//! A thread pool for parallel loops.
//!
//! A forward pass runs a few hundred short parallel loops per token, so the
//! cost of starting one matters more than anything else here. Workers spin
//! for a moment after each loop, which lets the next one start within
//! nanoseconds, and go to sleep only when no work arrives.
//!
//! Work is handed out in chunks claimed from a shared counter. A fast core
//! simply claims more chunks than a slow one, which suits CPUs that mix
//! performance and efficiency cores.

use std::cell::UnsafeCell;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// How long an idle worker spins before it goes to sleep.
const SPIN_TIME: Duration = Duration::from_micros(200);

/// A loop body: called with a chunk index and the slot of the thread that
/// runs it.
type Body<'a> = dyn Fn(usize, usize) + Sync + 'a;

/// The loop body of the job in progress, with its lifetime erased.
#[derive(Clone, Copy)]
struct Job {
    body: *const Body<'static>,
    chunks: usize,
}

/// State shared between the pool and its workers.
struct Shared {
    /// The current job. Written by the caller before `epoch` advances and
    /// read by workers after they observe the new epoch.
    job: UnsafeCell<Job>,
    /// Advances by one for every job.
    epoch: AtomicUsize,
    /// The next chunk index to claim.
    next: AtomicUsize,
    /// Workers that have not finished the current job.
    pending: AtomicUsize,
    /// Set when the pool is dropped.
    stop: AtomicBool,
    /// Set when a worker panicked. The pool refuses further jobs.
    poisoned: AtomicBool,
}

// SAFETY: `job` is only written while no worker is running a job, and the
// pointer inside it is only dereferenced while the caller of `run` keeps the
// body alive. Everything else is atomic.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// A fixed set of threads that run loop chunks in parallel.
pub struct ThreadPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// Serializes `run`, so the pool can be shared between sessions.
    busy: Mutex<()>,
}

impl ThreadPool {
    /// Creates a pool that runs loops on `threads` threads in total. The
    /// calling thread is one of them, so `threads - 1` workers are started.
    pub fn new(threads: usize) -> ThreadPool {
        fn idle(_: usize, _: usize) {}
        let idle: &'static Body<'static> = &idle;
        let shared = Arc::new(Shared {
            job: UnsafeCell::new(Job {
                body: idle,
                chunks: 0,
            }),
            epoch: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
            pending: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
        });
        let workers = (1..threads.max(1))
            .filter_map(|index| {
                let shared = Arc::clone(&shared);
                thread::Builder::new()
                    .name(format!("smallm-{index}"))
                    .spawn(move || worker(&shared, index))
                    .ok()
            })
            .collect();
        ThreadPool {
            shared,
            workers,
            busy: Mutex::new(()),
        }
    }

    /// Number of threads a loop runs on, counting the caller.
    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Calls `body(i)` once for every `i` in `0..chunks`, spread over the
    /// pool, and returns when all calls have finished.
    pub fn run(&self, chunks: usize, body: &(dyn Fn(usize) + Sync)) {
        self.run_slotted(chunks, &|index, _slot| body(index));
    }

    /// Like [`run`](Self::run), and also passes the slot of the thread that
    /// makes each call: a number below [`threads`](Self::threads) that no
    /// two threads share. The caller's own thread is slot 0.
    pub fn run_slotted(&self, chunks: usize, body: &Body<'_>) {
        if chunks <= 1 || self.workers.is_empty() {
            for index in 0..chunks {
                body(index, 0);
            }
            return;
        }
        let _busy = self
            .busy
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let shared = &*self.shared;
        assert!(
            !shared.poisoned.load(Ordering::Relaxed),
            "a worker thread panicked"
        );

        // SAFETY: no worker is inside a job: the previous `run` waited for
        // all of them. The lifetime of `body` is erased here and restored by
        // the wait below, which keeps this frame alive until every worker is
        // done with the pointer.
        unsafe {
            let body = std::mem::transmute::<&Body<'_>, *const Body<'static>>(body);
            *shared.job.get() = Job { body, chunks };
        }
        shared.next.store(0, Ordering::Relaxed);
        shared.pending.store(self.workers.len(), Ordering::Relaxed);
        shared.epoch.fetch_add(1, Ordering::Release);
        for worker in &self.workers {
            worker.thread().unpark();
        }

        // Waits for the workers even if `body` panics on this thread.
        struct WaitForWorkers<'a>(&'a Shared);
        impl Drop for WaitForWorkers<'_> {
            fn drop(&mut self) {
                let mut spins = 0u32;
                while self.0.pending.load(Ordering::Acquire) != 0 {
                    spins += 1;
                    if spins % 1024 == 0 {
                        thread::yield_now();
                    } else {
                        std::hint::spin_loop();
                    }
                }
            }
        }
        let wait = WaitForWorkers(shared);
        claim_chunks(shared, chunks, body, 0);
        drop(wait);
        assert!(
            !shared.poisoned.load(Ordering::Relaxed),
            "a worker thread panicked"
        );
    }

    /// Splits `0..n` into ranges of about `grain` items and calls `body` on
    /// each range in parallel. Runs on the calling thread when one range
    /// covers everything.
    pub fn ranges(&self, n: usize, grain: usize, body: &(dyn Fn(Range<usize>) + Sync)) {
        let grain = grain.max(1);
        self.run(n.div_ceil(grain), &|index| {
            body(index * grain..((index + 1) * grain).min(n))
        });
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        for worker in self.workers.drain(..) {
            worker.thread().unpark();
            let _ = worker.join();
        }
    }
}

/// Claims chunks from the shared counter until none are left.
fn claim_chunks(shared: &Shared, chunks: usize, body: &Body<'_>, slot: usize) {
    loop {
        let index = shared.next.fetch_add(1, Ordering::Relaxed);
        if index >= chunks {
            break;
        }
        body(index, slot);
    }
}

/// The loop of one worker thread: wait for a job, help with it, repeat.
fn worker(shared: &Shared, slot: usize) {
    let mut seen = 0;
    loop {
        // Wait for the epoch to advance. Spin first, then sleep.
        let mut idle_since: Option<Instant> = None;
        let mut spins = 0u32;
        loop {
            let epoch = shared.epoch.load(Ordering::Acquire);
            if epoch != seen {
                seen = epoch;
                break;
            }
            if shared.stop.load(Ordering::Acquire) {
                return;
            }
            spins += 1;
            if spins % 64 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let now = Instant::now();
            match idle_since {
                None => idle_since = Some(now),
                // `unpark` before `park` makes `park` return at once, so a
                // job posted between the check above and here is not missed.
                Some(start) if now.duration_since(start) >= SPIN_TIME => thread::park(),
                Some(_) => {}
            }
        }

        // Marks this worker done with the job, also when the body panics.
        struct Finish<'a>(&'a Shared);
        impl Drop for Finish<'_> {
            fn drop(&mut self) {
                if thread::panicking() {
                    self.0.poisoned.store(true, Ordering::Relaxed);
                }
                self.0.pending.fetch_sub(1, Ordering::Release);
            }
        }
        let finish = Finish(shared);
        // SAFETY: the epoch advanced, so `job` holds the current job, and the
        // caller of `run` keeps its body alive until `pending` reaches zero.
        let job = unsafe { *shared.job.get() };
        let body = unsafe { &*job.body };
        claim_chunks(shared, job.chunks, body, slot);
        drop(finish);
    }
}

/// A sensible default thread count for generating: one thread per
/// performance core.
///
/// Generation reads every weight once per token, so it waits on memory, and
/// slow cores only add threads that the fast ones then wait for. Matrix
/// kernels also gain nothing from two hardware threads sharing a core, so
/// logical CPUs are not counted twice.
pub fn default_threads() -> usize {
    let logical = logical_cpus();
    core_counts().map_or(logical, |cores| cores.fast.clamp(1, logical))
}

/// A sensible default thread count for reading prompts: one thread per
/// physical core, fast and slow alike.
///
/// A prompt is read in batches, where every weight serves many tokens, so
/// the work is arithmetic and every core adds to it. On an Intel Core
/// i7-14700K, 20 threads read a prompt 25 percent faster than its 8
/// performance cores alone. Hardware threads of a shared core made it
/// slower again.
pub fn default_batch_threads() -> usize {
    let logical = logical_cpus();
    core_counts().map_or(logical, |cores| cores.all.clamp(1, logical))
}

fn logical_cpus() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Physical cores: the fast ones, and all of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cores {
    fast: usize,
    all: usize,
}

/// Counts the cores in the fast group of a CPU, given one speed rating per
/// core. `None` when the ratings do not show two groups.
///
/// The CPU counts as mixed when its slowest core rates under 85 percent of
/// its fastest. Smaller gaps are the per-core turbo limits of an ordinary
/// CPU. The fast group is every core rated above the midpoint.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "android", test)),
    allow(dead_code)
)]
fn fast_cores(ratings: &[u64]) -> Option<usize> {
    let fastest = *ratings.iter().max()?;
    let slowest = *ratings.iter().min()?;
    if slowest == 0 || slowest.saturating_mul(100) >= fastest.saturating_mul(85) {
        return None;
    }
    let midpoint = slowest + (fastest - slowest) / 2;
    Some(ratings.iter().filter(|&&rating| rating > midpoint).count())
}

/// One logical CPU as the Linux kernel describes it in sysfs.
#[derive(Clone, Debug, Default)]
#[cfg_attr(
    not(any(target_os = "linux", target_os = "android", test)),
    allow(dead_code)
)]
struct LogicalCpu {
    /// The N of `cpuN`.
    id: usize,
    /// The logical CPUs that share this CPU's physical core, this one
    /// included: `topology/core_cpus_list`, or `thread_siblings_list` on
    /// older kernels. Empty when the kernel gives neither.
    siblings: Vec<usize>,
    /// Package, cluster and core ids, for kernels without a sibling list.
    package: String,
    cluster: String,
    core: String,
    /// `cpu_capacity`, which ARM kernels report. Zero when absent.
    capacity: u64,
    /// `cpufreq/cpuinfo_max_freq`. Zero when absent.
    max_freq: u64,
}

/// Counts physical cores among the logical CPUs a process may run on.
///
/// A physical core is named by the CPUs that share it, from the sibling
/// list, so that it is told apart from cores in other clusters. Core ids
/// alone are not enough: on some ARM boards each cluster numbers its cores
/// from zero again (the Allwinner A733's two A76 cores are cores 0 and 1 of
/// cluster 1), and those cores would be taken for hardware threads of the
/// first cluster's. Package, cluster and core id together serve when the
/// kernel gives no sibling list.
///
/// `allowed` is the process's CPU affinity. CPUs outside it are not
/// counted, so that a program confined to one cluster (`taskset -c 0-5`)
/// sizes its threads for that cluster. Each core is rated by its capacity,
/// or else by its top frequency, which tells the core types of an x86 CPU
/// apart.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "android", test)),
    allow(dead_code)
)]
fn count_cores(cpus: &[LogicalCpu], allowed: Option<&[usize]>) -> Option<Cores> {
    use std::collections::HashMap;

    // Capacity and top frequency of each physical core, by its name.
    let mut cores: HashMap<String, (u64, u64)> = HashMap::new();
    for cpu in cpus {
        if allowed.is_some_and(|allowed| !allowed.contains(&cpu.id)) {
            continue;
        }
        let name = match cpu.siblings.iter().min() {
            Some(first) => format!("cpu {first}"),
            None => format!("{} {} {}", cpu.package, cpu.cluster, cpu.core),
        };
        let rating = cores.entry(name).or_default();
        rating.0 = rating.0.max(cpu.capacity);
        rating.1 = rating.1.max(cpu.max_freq);
    }
    if cores.is_empty() {
        return None;
    }
    let capacities: Vec<u64> = cores.values().map(|rating| rating.0).collect();
    let frequencies: Vec<u64> = cores.values().map(|rating| rating.1).collect();
    let fast = fast_cores(&capacities)
        .or_else(|| fast_cores(&frequencies))
        .unwrap_or(cores.len());
    Some(Cores {
        fast,
        all: cores.len(),
    })
}

/// Parses a kernel CPU list such as `0-5,8,10-11`. `None` when the text is
/// not one.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "android", test)),
    allow(dead_code)
)]
fn parse_cpu_list(text: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in text.trim().split(',').filter(|part| !part.is_empty()) {
        match part.split_once('-') {
            Some((first, last)) => {
                let (first, last): (usize, usize) = (first.parse().ok()?, last.parse().ok()?);
                // A bound keeps a malformed range from allocating without end.
                if last < first || last - first > 1 << 16 {
                    return None;
                }
                cpus.extend(first..=last);
            }
            None => cpus.push(part.parse().ok()?),
        }
    }
    Some(cpus)
}

/// Counts cores from the kernel's CPU topology, among the CPUs this process
/// may run on.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn core_counts() -> Option<Cores> {
    use std::fs;

    let mut cpus = Vec::new();
    for entry in fs::read_dir("/sys/devices/system/cpu").ok()? {
        let path = entry.ok()?.path();
        let name = path.file_name()?.to_str()?;
        let Some(id) = name
            .strip_prefix("cpu")
            .and_then(|rest| rest.parse::<usize>().ok())
        else {
            continue;
        };
        let read = |file: &str| {
            fs::read_to_string(path.join(file))
                .ok()
                .map(|text| text.trim().to_string())
        };
        // A CPU that is offline has no topology directory.
        let (Some(package), Some(core)) = (
            read("topology/physical_package_id"),
            read("topology/core_id"),
        ) else {
            continue;
        };
        let number = |file: &str| read(file).and_then(|text| text.parse().ok()).unwrap_or(0);
        let siblings = read("topology/core_cpus_list")
            .or_else(|| read("topology/thread_siblings_list"))
            .and_then(|text| parse_cpu_list(&text))
            .unwrap_or_default();
        cpus.push(LogicalCpu {
            id,
            siblings,
            package,
            cluster: read("topology/cluster_id").unwrap_or_default(),
            core,
            capacity: number("cpu_capacity"),
            max_freq: number("cpufreq/cpuinfo_max_freq"),
        });
    }
    // The CPUs this process may run on, from its affinity mask.
    let allowed = fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
                .and_then(parse_cpu_list)
        });
    count_cores(&cpus, allowed.as_deref())
}

/// Counts performance cores on Apple silicon, and all physical cores.
#[cfg(target_os = "macos")]
fn core_counts() -> Option<Cores> {
    use std::ffi::{CStr, c_char, c_int, c_void};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            old: *mut c_void,
            old_len: *mut usize,
            new: *mut c_void,
            new_len: usize,
        ) -> c_int;
    }
    let read = |name: &CStr| {
        let mut value: c_int = 0;
        let mut size = std::mem::size_of::<c_int>();
        // SAFETY: `value` and `size` describe a writable integer, and the
        // name is a NUL-terminated string.
        let status = unsafe {
            sysctlbyname(
                name.as_ptr(),
                (&raw mut value).cast(),
                &raw mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        (status == 0 && value > 0).then_some(value as usize)
    };
    let all = read(c"hw.physicalcpu")?;
    // Intel Macs have one kind of core and no performance levels.
    let fast = read(c"hw.perflevel0.physicalcpu").unwrap_or(all);
    Some(Cores { fast, all })
}

/// Counts the physical cores, and those in the highest efficiency class,
/// which is how Windows names the fastest core type.
#[cfg(windows)]
fn core_counts() -> Option<Cores> {
    /// Asks for one record per physical core.
    const RELATION_PROCESSOR_CORE: u32 = 0;
    /// Byte offsets inside a record: its size, and the class of its core.
    const SIZE_AT: usize = 4;
    const CLASS_AT: usize = 9;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetLogicalProcessorInformationEx(
            relationship: u32,
            buffer: *mut u64,
            length: *mut u32,
        ) -> i32;
    }

    // The first call fails on purpose and reports the size the records need.
    let mut length = 0u32;
    // SAFETY: a null buffer of length zero is the documented size query.
    unsafe {
        GetLogicalProcessorInformationEx(
            RELATION_PROCESSOR_CORE,
            std::ptr::null_mut(),
            &raw mut length,
        )
    };
    if length == 0 {
        return None;
    }
    // 64-bit words keep the records aligned the way the system writes them.
    let mut words = vec![0u64; (length as usize).div_ceil(8)];
    // SAFETY: the buffer holds at least `length` writable bytes.
    let filled = unsafe {
        GetLogicalProcessorInformationEx(
            RELATION_PROCESSOR_CORE,
            words.as_mut_ptr(),
            &raw mut length,
        )
    };
    if filled == 0 {
        return None;
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
    let bytes = bytes.get(..length as usize)?;

    // Records vary in size and each one says how long it is.
    let mut classes = Vec::new();
    let mut offset = 0;
    while let Some(record) = bytes.get(offset..).filter(|rest| rest.len() > CLASS_AT) {
        let size = u32::from_ne_bytes(record[SIZE_AT..SIZE_AT + 4].try_into().ok()?) as usize;
        if size <= CLASS_AT {
            break;
        }
        classes.push(record[CLASS_AT]);
        offset += size;
    }
    let fastest = *classes.iter().max()?;
    Some(Cores {
        fast: classes.iter().filter(|&&class| class == fastest).count(),
        all: classes.len(),
    })
}

/// Elsewhere the topology is not read. Most x86 CPUs run two hardware
/// threads per core, so half the logical count is the closer guess there.
#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    windows
)))]
fn core_counts() -> Option<Cores> {
    let logical = logical_cpus();
    let physical = if cfg!(any(target_arch = "x86", target_arch = "x86_64")) && logical >= 4 {
        logical / 2
    } else {
        logical
    };
    Some(Cores {
        fast: physical,
        all: physical,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn every_chunk_runs_exactly_once() {
        let pool = ThreadPool::new(4);
        assert_eq!(pool.threads(), 4);
        for chunks in [0, 1, 2, 3, 17, 1000] {
            let counts: Vec<AtomicUsize> = (0..chunks).map(|_| AtomicUsize::new(0)).collect();
            pool.run(chunks, &|index| {
                counts[index].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                counts
                    .iter()
                    .all(|count| count.load(Ordering::Relaxed) == 1),
                "{chunks} chunks"
            );
        }
    }

    #[test]
    fn ranges_cover_everything_without_overlap() {
        let pool = ThreadPool::new(3);
        for (n, grain) in [
            (0, 4),
            (1, 4),
            (10, 3),
            (64, 64),
            (65, 64),
            (1000, 7),
            (5, 0),
        ] {
            let sum = AtomicU64::new(0);
            let calls = AtomicUsize::new(0);
            pool.ranges(n, grain, &|range| {
                calls.fetch_add(1, Ordering::Relaxed);
                sum.fetch_add(range.map(|i| i as u64).sum(), Ordering::Relaxed);
            });
            assert_eq!(sum.load(Ordering::Relaxed), (0..n as u64).sum::<u64>());
            assert_eq!(calls.load(Ordering::Relaxed), n.div_ceil(grain.max(1)));
        }
    }

    #[test]
    fn a_single_thread_pool_runs_inline() {
        let pool = ThreadPool::new(1);
        let caller = thread::current().id();
        pool.run(8, &|_| assert_eq!(thread::current().id(), caller));
        let pool = ThreadPool::new(0);
        assert_eq!(pool.threads(), 1);
    }

    #[test]
    fn many_short_jobs_and_sleeping_workers_stay_correct() {
        let pool = ThreadPool::new(4);
        let total = AtomicU64::new(0);
        for round in 0..20_000u64 {
            pool.run(4, &|index| {
                total.fetch_add(round + index as u64, Ordering::Relaxed);
            });
            // Let the workers fall asleep now and then.
            if round % 5_000 == 0 {
                thread::sleep(Duration::from_millis(2));
            }
        }
        let expected: u64 = (0..20_000u64).map(|round| 4 * round + 6).sum();
        assert_eq!(total.load(Ordering::Relaxed), expected);
    }

    #[test]
    fn a_pool_can_be_shared_between_threads() {
        let pool = Arc::new(ThreadPool::new(3));
        let total = Arc::new(AtomicU64::new(0));
        let callers: Vec<_> = (0..4)
            .map(|_| {
                let (pool, total) = (Arc::clone(&pool), Arc::clone(&total));
                thread::spawn(move || {
                    for _ in 0..500 {
                        pool.run(6, &|index| {
                            total.fetch_add(index as u64, Ordering::Relaxed);
                        });
                    }
                })
            })
            .collect();
        for caller in callers {
            caller.join().unwrap();
        }
        assert_eq!(total.load(Ordering::Relaxed), 4 * 500 * 15);
    }

    #[test]
    fn the_body_may_borrow_from_the_caller() {
        let pool = ThreadPool::new(4);
        let input: Vec<u64> = (0..10_000).collect();
        let mut output = vec![0u64; input.len()];
        let out = output.as_mut_ptr() as usize;
        pool.ranges(input.len(), 100, &|range| {
            for i in range {
                // SAFETY: ranges do not overlap, so each element is written
                // by one thread.
                unsafe { *(out as *mut u64).add(i) = input[i] * 2 };
            }
        });
        assert!(
            output
                .iter()
                .enumerate()
                .all(|(i, value)| *value == 2 * i as u64)
        );
    }

    #[test]
    fn slots_are_unique_per_thread() {
        let pool = ThreadPool::new(4);
        let owners: Vec<Mutex<Option<thread::ThreadId>>> =
            (0..4).map(|_| Mutex::new(None)).collect();
        for _ in 0..200 {
            pool.run_slotted(64, &|_, slot| {
                let mut owner = owners[slot].lock().unwrap();
                let me = thread::current().id();
                assert_eq!(*owner.get_or_insert(me), me, "slot {slot} is shared");
            });
        }
        // Slot 0 is the caller's. The workers can take every chunk before
        // the caller gets to one, so the slot may also have gone unused.
        let first = *owners[0].lock().unwrap();
        assert!(first.is_none() || first == Some(thread::current().id()));
    }

    #[test]
    fn fast_cores_are_told_from_slow_ones() {
        // An Intel CPU with 8 fast and 12 slow cores, by top frequency.
        let mut intel = vec![5_600_000; 2];
        intel.extend([5_500_000; 6]);
        intel.extend([4_300_000; 12]);
        assert_eq!(fast_cores(&intel), Some(8));
        // An ARM board with 4 big and 4 little cores, by capacity.
        assert_eq!(
            fast_cores(&[1024, 1024, 1024, 1024, 414, 414, 414, 414]),
            Some(4)
        );
        // Three kinds of core: the middle kind counts as fast.
        assert_eq!(
            fast_cores(&[1024, 870, 870, 870, 870, 350, 350, 350]),
            Some(5)
        );
        // Equal cores, and cores that differ only by their turbo limits.
        assert_eq!(fast_cores(&[1024; 8]), None);
        assert_eq!(
            fast_cores(&[5_900_000, 5_750_000, 5_600_000, 5_450_000]),
            None
        );
        // No ratings, or a core without one.
        assert_eq!(fast_cores(&[]), None);
        assert_eq!(fast_cores(&[0, 0, 0, 0]), None);
        assert_eq!(fast_cores(&[5_500_000, 0]), None);
    }

    /// A logical CPU of a test topology.
    fn cpu(id: usize, siblings: &[usize], cluster: &str, core: &str, capacity: u64) -> LogicalCpu {
        LogicalCpu {
            id,
            siblings: siblings.to_vec(),
            package: "0".into(),
            cluster: cluster.into(),
            core: core.into(),
            capacity,
            max_freq: 0,
        }
    }

    #[test]
    fn clusters_that_number_their_cores_from_zero_are_told_apart() {
        // The Allwinner A733: six A55s are cores 0 to 5 of cluster 0, and
        // two A76s are cores 0 and 1 of cluster 1.
        let mut a733: Vec<LogicalCpu> = (0..6)
            .map(|i| cpu(i, &[i], "0", &i.to_string(), 446))
            .collect();
        a733.push(cpu(6, &[6], "1", "0", 1024));
        a733.push(cpu(7, &[7], "1", "1", 1024));
        assert_eq!(count_cores(&a733, None), Some(Cores { fast: 2, all: 8 }));
        // Without sibling lists, the cluster id tells them apart.
        let mut old_kernel = a733.clone();
        old_kernel.iter_mut().for_each(|cpu| cpu.siblings.clear());
        assert_eq!(
            count_cores(&old_kernel, None),
            Some(Cores { fast: 2, all: 8 })
        );
        // Confined to the A55s, a process sees six equal cores.
        let little = [0, 1, 2, 3, 4, 5];
        assert_eq!(
            count_cores(&a733, Some(&little)),
            Some(Cores { fast: 6, all: 6 })
        );
        assert_eq!(
            count_cores(&a733, Some(&[6, 7])),
            Some(Cores { fast: 2, all: 2 })
        );
        assert_eq!(count_cores(&a733, Some(&[])), None);
    }

    #[test]
    fn hardware_threads_of_one_core_count_once() {
        // Two cores with two hardware threads each, numbered as Intel does:
        // CPUs 0 and 2 share core 0, CPUs 1 and 3 share core 1.
        let cpus = [
            cpu(0, &[0, 2], "", "0", 0),
            cpu(1, &[1, 3], "", "1", 0),
            cpu(2, &[0, 2], "", "0", 0),
            cpu(3, &[1, 3], "", "1", 0),
        ];
        assert_eq!(count_cores(&cpus, None), Some(Cores { fast: 2, all: 2 }));
        // One thread of a core is enough for the core to count.
        assert_eq!(
            count_cores(&cpus, Some(&[2, 3])),
            Some(Cores { fast: 2, all: 2 })
        );
    }

    #[test]
    fn cpu_lists_parse() {
        assert_eq!(
            parse_cpu_list("0-5,8,10-11\n"),
            Some(vec![0, 1, 2, 3, 4, 5, 8, 10, 11])
        );
        assert_eq!(parse_cpu_list("7"), Some(vec![7]));
        assert_eq!(parse_cpu_list(""), Some(vec![]));
        assert_eq!(parse_cpu_list("3-1"), None);
        assert_eq!(parse_cpu_list("a"), None);
        assert_eq!(parse_cpu_list("0-999999999"), None);
    }

    #[test]
    fn default_thread_count_is_usable() {
        let threads = default_threads();
        assert!(threads >= 1);
        assert!(
            threads
                <= thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
        );
    }
}
