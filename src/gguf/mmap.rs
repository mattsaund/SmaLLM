//! A read-only view of a model file.
//!
//! The file is memory mapped where the platform allows it. Mapped pages are
//! shared with the page cache, so loading is instant and the weights cost no
//! private memory. Platforms without a mapping call read the file into the
//! heap instead.

use std::fs::File;
use std::path::Path;

use crate::error::Result;

/// The bytes of a model file, mapped or owned.
pub struct Mapping {
    ptr: *const u8,
    len: usize,
    backing: Backing,
}

/// What keeps the bytes alive, and how to release them.
enum Backing {
    /// A memory mapping made by the operating system.
    #[cfg(any(unix, windows))]
    Mapped,
    /// A heap buffer. Also used for models built in memory.
    Heap(#[allow(dead_code)] Vec<u8>),
}

// The mapping is read-only for its whole life, so sharing it is safe.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    /// Opens `path` read-only.
    pub fn open(path: &Path) -> Result<Mapping> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        if len > usize::MAX as u64 {
            return Err(crate::Error::invalid(
                "the file is larger than this system can address",
            ));
        }
        let len = len as usize;
        if len == 0 {
            return Ok(Mapping::from_vec(Vec::new()));
        }
        Mapping::map_file(file, len)
    }

    /// Wraps bytes that are already in memory.
    pub fn from_vec(bytes: Vec<u8>) -> Mapping {
        Mapping {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
            backing: Backing::Heap(bytes),
        }
    }

    /// The whole file.
    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr` points at `len` readable bytes for as long as `self`
        // lives, and nothing writes to them.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Size of the file in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the file has no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// True when the bytes are shared with the page cache.
    pub fn is_mapped(&self) -> bool {
        !matches!(self.backing, Backing::Heap(_))
    }

    #[cfg(unix)]
    fn map_file(file: File, len: usize) -> Result<Mapping> {
        use std::os::fd::AsRawFd;
        // SAFETY: a private read-only mapping of an open file. The kernel
        // validates every argument, and failure is reported as MAP_FAILED.
        let ptr = unsafe {
            sys::mmap(
                std::ptr::null_mut(),
                len,
                sys::PROT_READ,
                sys::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr as isize == -1 {
            // Some filesystems cannot be mapped. Reading still works there.
            return Mapping::read_file(file, len);
        }
        Ok(Mapping {
            ptr: ptr as *const u8,
            len,
            backing: Backing::Mapped,
        })
    }

    #[cfg(windows)]
    fn map_file(file: File, len: usize) -> Result<Mapping> {
        use std::os::windows::io::AsRawHandle;
        // SAFETY: both calls validate their arguments and return null on
        // failure. The view keeps the file data alive after the mapping
        // handle is closed.
        unsafe {
            let mapping = sys::CreateFileMappingW(
                file.as_raw_handle() as *mut _,
                std::ptr::null_mut(),
                sys::PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            );
            if mapping.is_null() {
                return Mapping::read_file(file, len);
            }
            let view = sys::MapViewOfFile(mapping, sys::FILE_MAP_READ, 0, 0, 0);
            sys::CloseHandle(mapping);
            if view.is_null() {
                return Mapping::read_file(file, len);
            }
            Ok(Mapping {
                ptr: view as *const u8,
                len,
                backing: Backing::Mapped,
            })
        }
    }

    #[cfg(not(any(unix, windows)))]
    fn map_file(file: File, len: usize) -> Result<Mapping> {
        Mapping::read_file(file, len)
    }

    /// Reads the whole file into the heap.
    fn read_file(mut file: File, len: usize) -> Result<Mapping> {
        use std::io::Read;
        let mut bytes = Vec::with_capacity(len);
        file.read_to_end(&mut bytes)?;
        Ok(Mapping::from_vec(bytes))
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        match self.backing {
            #[cfg(unix)]
            Backing::Mapped => {
                // SAFETY: `ptr` and `len` are exactly what mmap returned.
                unsafe {
                    sys::munmap(self.ptr as *mut _, self.len);
                }
            }
            #[cfg(windows)]
            Backing::Mapped => {
                // SAFETY: `ptr` is the address MapViewOfFile returned.
                unsafe {
                    sys::UnmapViewOfFile(self.ptr as *const _);
                }
            }
            Backing::Heap(_) => {}
        }
    }
}

/// The few system calls a mapping needs, declared here to avoid a dependency.
#[cfg(unix)]
mod sys {
    use std::ffi::{c_int, c_void};

    pub const PROT_READ: c_int = 1;
    pub const MAP_PRIVATE: c_int = 2;

    /// `off_t`. It is 32 bits wide only on 32-bit systems whose C library
    /// predates large file support by default.
    #[cfg(all(
        target_pointer_width = "32",
        any(target_env = "gnu", target_env = "uclibc", target_os = "android")
    ))]
    pub type Offset = i32;
    #[cfg(not(all(
        target_pointer_width = "32",
        any(target_env = "gnu", target_env = "uclibc", target_os = "android")
    )))]
    pub type Offset = i64;

    unsafe extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: Offset,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> c_int;
    }
}

/// The few Win32 calls a mapping needs, declared here to avoid a dependency.
#[cfg(windows)]
mod sys {
    use std::ffi::c_void;

    pub const PAGE_READONLY: u32 = 0x02;
    pub const FILE_MAP_READ: u32 = 0x04;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn CreateFileMappingW(
            file: *mut c_void,
            attributes: *mut c_void,
            protect: u32,
            size_high: u32,
            size_low: u32,
            name: *const u16,
        ) -> *mut c_void;
        pub fn MapViewOfFile(
            mapping: *mut c_void,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            bytes: usize,
        ) -> *mut c_void;
        pub fn UnmapViewOfFile(address: *const c_void) -> i32;
        pub fn CloseHandle(handle: *mut c_void) -> i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_mapping_exposes_its_bytes() {
        let map = Mapping::from_vec(vec![1, 2, 3]);
        assert_eq!(map.as_slice(), &[1, 2, 3]);
        assert_eq!(map.len(), 3);
        assert!(!map.is_mapped());
    }

    #[test]
    fn file_mapping_matches_the_file() {
        let path = std::env::temp_dir().join(format!("smallm-mmap-test-{}", std::process::id()));
        std::fs::write(&path, b"hello mapping").unwrap();
        let map = Mapping::open(&path).unwrap();
        assert_eq!(map.as_slice(), b"hello mapping");
        drop(map);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn empty_file_is_an_empty_slice() {
        let path = std::env::temp_dir().join(format!("smallm-mmap-empty-{}", std::process::id()));
        std::fs::write(&path, b"").unwrap();
        let map = Mapping::open(&path).unwrap();
        assert!(map.is_empty());
        drop(map);
        std::fs::remove_file(&path).unwrap();
    }
}
