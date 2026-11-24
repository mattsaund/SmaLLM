# Install SmaLLM on Windows.
#
#   irm https://raw.githubusercontent.com/mattsaund/SmaLLM/main/install.ps1 | iex
#
# Downloads the ready-made smallm.exe from the latest GitHub release and
# checks it against the release's SHA-256 list. When there is none to
# download, it builds from source. Rust is installed for that, after
# asking, with a toolchain that needs nothing else on x86-64. Without git,
# the source comes as an archive.
#
# Every step prints a line, and one progress bar for the whole installation
# stays at the bottom of the window.
#
# To remove SmaLLM again, with its downloaded models:
#
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/mattsaund/SmaLLM/main/install.ps1))) -Uninstall
#
# `smallm uninstall` does the same. Add -Yes to remove without being asked.
#
# Override anything with the environment:
#   $env:SMALLM_REPO         git URL (default: https://github.com/mattsaund/SmaLLM.git)
#   $env:SMALLM_REF          branch or tag (default: main, the newest release).
#                            A version tag like v0.1.0 installs that release
#   $env:SMALLM_PREFIX       where smallm.exe lands (default: %USERPROFILE%\.local\bin)
#   $env:SMALLM_FROM_SOURCE  set to anything to build instead of downloading
#
# Three rules this file keeps:
#
#   * It is plain ASCII. Windows PowerShell 5.1 reads a local .ps1 in the ANSI
#     code page, where a byte of a multi-byte character can end a string.
#
#   * It never sets $ErrorActionPreference to Stop. 5.1 turns every line a
#     native program writes to stderr into an error, and git and cargo write
#     their progress there. Failures are checked where they happen instead.
#
#   * It never calls exit. Piped through iex, this runs in the caller's own
#     session, and an exit there closes the window.

param([switch]$Uninstall, [switch]$Yes)

& {
    param([bool]$Uninstall, [bool]$Yes)
    $ErrorActionPreference = 'Continue'

    # GitHub only speaks TLS 1.2 and up, and Windows PowerShell 5.1 on an
    # older Windows does not offer it unless asked.
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    # --- the screen ----------------------------------------------------------
    #
    # A log that scrolls, and one progress bar under it. Write-Log clears the
    # bar, prints its line, and draws the bar again, so the bar stays on the
    # last line however much is printed above it. Without a console, as in
    # CI or with the output redirected, there is no bar, only the log.
    #
    # The state lives in one table, so the functions below can change it
    # without reaching into scopes that are not theirs.

    $screen = @{
        Live     = $false
        Percent  = 0
        Activity = 'starting'
        Work     = $null
    }
    try {
        $screen.Live = -not [Console]::IsOutputRedirected -and [Console]::WindowWidth -gt 20
    }
    catch {
        $screen.Live = $false
    }

    function Get-Width {
        try { return [Math]::Max(40, [Console]::WindowWidth - 1) } catch { return 79 }
    }

    # Takes the bar off the screen.
    function Clear-Bar {
        if ($screen.Live) {
            [Console]::Write("`r" + (' ' * (Get-Width)) + "`r")
        }
    }

    # Draws the bar on the last line, over the one that was there.
    function Show-Bar {
        if (-not $screen.Live) { return }
        $width = 30
        $filled = [int][Math]::Floor($screen.Percent * $width / 100)
        $line = '[' + ('#' * $filled) + ('-' * ($width - $filled)) + '] ' +
            ('{0,3}%' -f $screen.Percent) + '  ' + $screen.Activity
        $room = Get-Width
        if ($line.Length -gt $room) { $line = $line.Substring(0, $room) }
        [Console]::Write("`r" + $line.PadRight($room) + "`r" + $line)
    }

    # One line of the log. The tag is a word like Downloading.
    function Write-Log([string]$Tag, [string]$Text, [string]$Color = 'Green') {
        Clear-Bar
        Write-Host ('{0,12}' -f $Tag) -ForegroundColor $Color -NoNewline
        Write-Host (' ' + $Text)
        Show-Bar
    }

    # A log line about something that did not stop the installation.
    function Write-Warn([string]$Tag, [string]$Text) {
        Write-Log $Tag $Text 'Yellow'
    }

    function Set-Progress([int]$Percent, [string]$Activity) {
        $screen.Percent = [Math]::Max(0, [Math]::Min(100, $Percent))
        $screen.Activity = $Activity
        Show-Bar
    }

    # A byte count for people: 1.2 MB.
    function Format-Bytes([double]$Bytes) {
        if ($Bytes -ge 1e9) { return '{0:N1} GB' -f ($Bytes / 1e9) }
        if ($Bytes -ge 1e6) { return '{0:N1} MB' -f ($Bytes / 1e6) }
        if ($Bytes -ge 1e3) { return '{0:N1} KB' -f ($Bytes / 1e3) }
        return '{0} B' -f [int]$Bytes
    }

    # Seconds as minutes and seconds: 1:05.
    function Format-Clock([int]$Seconds) {
        return '{0}:{1:00}' -f [Math]::Floor($Seconds / 60), ($Seconds % 60)
    }

    function Have([string]$Name) {
        [bool](Get-Command $Name -ErrorAction SilentlyContinue)
    }

    # --- steps that both routes share ----------------------------------------

    # Downloads $Url to $Path while the bar moves from $From to $To percent.
    # Returns whether it worked. Reads the response as it arrives, so the
    # bar follows the bytes.
    function Get-File([string]$Url, [string]$Path, [int]$From, [int]$To) {
        Add-Type -AssemblyName System.Net.Http
        $client = New-Object System.Net.Http.HttpClient
        $name = Split-Path $Path -Leaf
        try {
            $response = $client.GetAsync($Url, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
            if (-not $response.IsSuccessStatusCode) { return $false }
            $total = $response.Content.Headers.ContentLength
            if ($total) {
                Write-Log 'Downloading' ('{0} ({1})' -f $name, (Format-Bytes $total))
            }
            else {
                Write-Log 'Downloading' $name
            }
            $source = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
            $target = [System.IO.File]::Create($Path)
            try {
                $buffer = New-Object byte[] 65536
                $got = 0
                $drawn = [DateTime]::MinValue
                while (($read = $source.Read($buffer, 0, $buffer.Length)) -gt 0) {
                    $target.Write($buffer, 0, $read)
                    $got += $read
                    if (([DateTime]::Now - $drawn).TotalMilliseconds -ge 100) {
                        $drawn = [DateTime]::Now
                        if ($total) {
                            $percent = $From + [int](($To - $From) * $got / $total)
                            Set-Progress $percent ('downloading {0}  {1} / {2}' -f $name, (Format-Bytes $got), (Format-Bytes $total))
                        }
                        else {
                            Set-Progress $From ('downloading {0}  {1}' -f $name, (Format-Bytes $got))
                        }
                    }
                }
            }
            finally {
                $target.Dispose()
                $source.Dispose()
            }
            Write-Log 'Received' ('{0}, {1}' -f $name, (Format-Bytes $got))
            Set-Progress $To ('downloaded ' + $name)
            return $true
        }
        catch {
            return $false
        }
        finally {
            $client.Dispose()
        }
    }

    # One line that a tool printed, as a line of the log. cargo already
    # writes a tag and its text.
    function Show-ToolLine([string]$Line) {
        $text = $Line.Trim()
        if (-not $text) { return }
        if ($text -match '^warning:\s*(.*)$') { Write-Warn 'Warning' $Matches[1]; return }
        if ($text -match '^error(\[\w+\])?:\s*(.*)$') { Write-Warn 'Error' $Matches[2]; return }
        $word = ($text -split ' ', 2)[0]
        $known = @('Compiling', 'Finished', 'Downloading', 'Downloaded', 'Updating', 'Locking', 'Building', 'Checking', 'Fresh')
        if ($known -contains $word) {
            Write-Log $word ($text.Substring($word.Length).Trim())
        }
        else {
            Write-Log '' $text
        }
    }

    # Reads what was added to $Path since $State.Offset, and shows the whole
    # lines. A line still being written waits in $State.Partial.
    function Show-NewLines([string]$Path, [hashtable]$State, [bool]$Final) {
        if (-not (Test-Path $Path)) { return }
        $stream = [System.IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite')
        try {
            [void]$stream.Seek($State.Offset, 'Begin')
            $length = $stream.Length - $State.Offset
            if ($length -le 0 -and -not $Final) { return }
            $bytes = New-Object byte[] ([Math]::Max(0, $length))
            $read = $stream.Read($bytes, 0, $bytes.Length)
            $State.Offset += $read
            $text = $State.Partial + [System.Text.Encoding]::UTF8.GetString($bytes, 0, $read)
        }
        finally {
            $stream.Dispose()
        }
        $lines = $text -split "`r?`n|`r"
        if ($Final) {
            $State.Partial = ''
        }
        else {
            $State.Partial = $lines[-1]
            $lines = if ($lines.Count -gt 1) { $lines[0..($lines.Count - 2)] } else { @() }
        }
        foreach ($line in $lines) { Show-ToolLine $line }
    }

    # Runs a program, shows each line it prints in the log, and moves the bar
    # from $From toward $To while it runs. Nothing measures how far along a
    # build is, so the bar closes in on $To and slows down as it does: it is
    # halfway there after a minute. Returns the program's exit code.
    function Invoke-Logged([int]$From, [int]$To, [string]$Activity, [string]$File, [string[]]$Arguments) {
        $out = Join-Path $screen.Work 'step.out'
        $err = Join-Path $screen.Work 'step.err'
        Remove-Item $out, $err -Force -ErrorAction SilentlyContinue
        # Start-Process joins the arguments with spaces, so any with a space
        # in it is quoted here.
        $quoted = $Arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
        $process = Start-Process -FilePath $File -ArgumentList $quoted -NoNewWindow -PassThru `
            -RedirectStandardOutput $out -RedirectStandardError $err
        # Reading the handle now keeps the exit code readable later, which
        # PowerShell otherwise loses for processes started this way.
        $null = $process.Handle
        $states = @{ $out = @{ Offset = 0L; Partial = '' }; $err = @{ Offset = 0L; Partial = '' } }
        $started = Get-Date
        while ($true) {
            $finished = $process.HasExited
            foreach ($path in @($err, $out)) { Show-NewLines $path $states[$path] $finished }
            if ($finished) { break }
            $elapsed = [int]((Get-Date) - $started).TotalSeconds
            Set-Progress ($From + [int](($To - $From) * $elapsed / ($elapsed + 60))) ('{0} ({1})' -f $Activity, (Format-Clock $elapsed))
            Start-Sleep -Milliseconds 200
        }
        $process.WaitForExit()
        return $process.ExitCode
    }

    # Copy $From to $Prefix\smallm.exe, over any smallm.exe already there.
    # Windows will not overwrite a program while it runs, but it will rename
    # one, so a running copy is moved aside first.
    function Install-Exe([string]$From, [string]$Prefix) {
        New-Item -ItemType Directory -Force -Path $Prefix | Out-Null
        $dest = Join-Path $Prefix 'smallm.exe'
        Get-ChildItem -Path $Prefix -Filter 'smallm.exe.old-*' -ErrorAction SilentlyContinue |
            Remove-Item -Force -ErrorAction SilentlyContinue
        if (Test-Path $dest) {
            Remove-Item -Path $dest -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path $dest) {
            $aside = Join-Path $Prefix ('smallm.exe.old-' + [Guid]::NewGuid().ToString('N').Substring(0, 8))
            Move-Item -Path $dest -Destination $aside -ErrorAction Stop
        }
        Copy-Item -Path $From -Destination $dest -Force -ErrorAction Stop
    }

    # Whether this is Arm64 Windows. Asked of the machine as well as of this
    # process, which may be an x64 PowerShell running under emulation.
    function Test-Arm64 {
        if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') {
            return $true
        }
        $machine = Get-ItemProperty -ErrorAction SilentlyContinue `
            -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment'
        return [bool]($machine -and $machine.PROCESSOR_ARCHITECTURE -eq 'ARM64')
    }

    # --- a ready-made binary ---------------------------------------------------

    # The release's zip for this machine, checked and unpacked into $Prefix.
    # Returns whether it worked. A missing release is a reason to build
    # instead. A checksum that does not match is a reason to stop.
    function Install-Prebuilt([string]$Repo, [string]$Ref, [string]$Prefix) {
        $slug = $Repo -replace '^https://github\.com/', '' -replace '\.git$', ''
        if ($slug -match '://' -or $slug -notmatch '^[^/]+/[^/]+$') { return $false }
        $release = if ($Ref -eq 'main') { 'latest/download' } else { "download/$Ref" }
        $base = "https://github.com/$slug/releases/$release"
        $target = if (Test-Arm64) { 'aarch64-pc-windows-msvc' } else { 'x86_64-pc-windows-msvc' }
        $asset = "smallm-$target.zip"
        $work = $screen.Work

        Write-Log 'Looking' "for a ready-made smallm for $target"
        Set-Progress 5 'looking for a release'
        $zip = Join-Path $work $asset
        if (-not (Get-File "$base/$asset" $zip 10 85)) {
            Write-Warn 'Missing' 'no release has one for this machine, so it will be built from source'
            return $false
        }
        $sums = Join-Path $work 'SHA256SUMS'
        if (Get-File "$base/SHA256SUMS" $sums 85 90) {
            $actual = (Get-FileHash -Algorithm SHA256 -Path $zip).Hash.ToLower()
            $line = Get-Content $sums | Where-Object { $_ -match ([regex]::Escape($asset) + '$') } | Select-Object -First 1
            $expected = if ($line) { ($line -split '\s+')[0].ToLower() } else { '' }
            if ($expected -ne $actual) {
                throw 'the download does not match its checksum. Nothing was installed.'
            }
            Write-Log 'Verified' ('SHA-256 {0}... matches the release' -f $actual.Substring(0, 16))
        }
        else {
            Write-Warn 'Unverified' 'no checksum to compare with'
        }
        Set-Progress 93 'unpacking'
        Expand-Archive -Path $zip -DestinationPath $work -Force -ErrorAction Stop
        $exe = Get-ChildItem -Path $work -Filter 'smallm.exe' -Recurse | Select-Object -First 1
        if (-not $exe) {
            Write-Warn 'Failed' 'the download had no smallm.exe in it, so it will be built from source'
            return $false
        }
        Write-Log 'Unpacked' ('smallm.exe, {0}' -f (Format-Bytes $exe.Length))
        Set-Progress 97 'installing'
        Install-Exe -From $exe.FullName -Prefix $Prefix
        return $true
    }

    # A yes-or-no question, where Enter means yes.
    function Confirm-Step([string]$Question) {
        Clear-Bar
        $answer = Read-Host ('{0,12} {1} [Y/n]' -f '', $Question)
        return -not ($answer -match '^\s*[Nn]')
    }

    # Installs Rust for the current user. On x86-64 the GNU toolchain is
    # chosen, which brings its own linker, so no Visual Studio is needed.
    function Install-Rust {
        Write-Warn 'Missing' 'SmaLLM is written in Rust, and cargo is not installed.'
        Write-Log '' "The Rust toolchain goes in ~\.rustup and ~\.cargo, and 'rustup self uninstall' removes it again."
        if (-not (Confirm-Step 'Install it now?')) {
            throw 'building from source needs Rust. Install it from https://rustup.rs and run this again'
        }
        $arm = Test-Arm64
        $machine = if ($arm) { 'aarch64' } else { 'x86_64' }
        $installer = Join-Path $screen.Work 'rustup-init.exe'
        if (-not (Get-File "https://win.rustup.rs/$machine" $installer 5 8)) {
            throw 'could not download the Rust installer'
        }
        $arguments = @('-y', '--profile', 'minimal', '--default-toolchain', 'stable')
        if (-not $arm) { $arguments += @('--default-host', 'x86_64-pc-windows-gnu') }
        $code = Invoke-Logged 8 40 'installing Rust' $installer $arguments
        if ($code -ne 0) { throw 'the Rust installer failed. The lines above say why' }
        if ($arm) {
            Write-Warn 'Note' 'on ARM64, Rust links with the Visual Studio Build Tools, which must be installed.'
        }
        $env:PATH = (Join-Path $HOME '.cargo\bin') + ";$env:PATH"
    }

    # --- build from source -----------------------------------------------------

    # Build with cargo from a checkout or a fresh clone.
    function Install-FromSource([string]$Repo, [string]$Ref, [string]$Prefix, [bool]$Checkout) {
        if (-not (Have 'cargo')) {
            $cargoBin = Join-Path $HOME '.cargo\bin'
            if (Test-Path (Join-Path $cargoBin 'cargo.exe')) {
                $env:PATH = "$cargoBin;$env:PATH"
            }
            else {
                Install-Rust
            }
        }
        Write-Log 'Found' ((& cargo --version) -join ' ')
        if ($Checkout) {
            $source = (Get-Location).Path
            Write-Log 'Building' "the checkout in $source"
        }
        else {
            $source = Join-Path $screen.Work 'src'
            if (Have 'git') {
                $code = Invoke-Logged 42 48 'fetching the source' 'git' @('clone', '--depth', '1', '--branch', $Ref, $Repo, $source)
                if ($code -ne 0) { throw "could not clone $Repo ($Ref)" }
            }
            else {
                # Without git, GitHub serves the same files as an archive.
                if ($Repo -notmatch '^https://github\.com/') { throw "git is needed to fetch $Repo" }
                $slug = $Repo -replace '^https://github\.com/', '' -replace '\.git$', ''
                $archive = Join-Path $screen.Work 'source.zip'
                if (-not (Get-File "https://github.com/$slug/archive/$Ref.zip" $archive 42 46)) {
                    throw 'could not download the source'
                }
                $unpacked = Join-Path $screen.Work 'unpacked'
                Expand-Archive -Path $archive -DestinationPath $unpacked -Force
                $top = Get-ChildItem -Path $unpacked -Directory | Select-Object -First 1
                if (-not $top) { throw 'the source archive was empty' }
                Move-Item -Path $top.FullName -Destination $source
            }
            $count = (Get-ChildItem -Path $source -Recurse -File | Where-Object { $_.FullName -notmatch '\\\.git\\' }).Count
            Write-Log 'Fetched' ('{0} ({1}), {2} files' -f $Repo, $Ref, $count)
        }
        $manifest = Join-Path $source 'Cargo.toml'
        $code = Invoke-Logged 50 97 'compiling' 'cargo' @('build', '--release', '--locked', '--manifest-path', $manifest, '--bin', 'smallm')
        if ($code -ne 0) { throw 'the build failed. The lines above say why' }
        $exe = Join-Path $source 'target\release\smallm.exe'
        if (-not (Test-Path $exe)) { throw 'the build finished without leaving a smallm.exe' }
        Write-Log 'Built' ('smallm.exe, {0}' -f (Format-Bytes (Get-Item $exe).Length))
        Set-Progress 98 'installing'
        Install-Exe -From $exe -Prefix $Prefix
    }

    # Put $Prefix on the user's PATH, for good and for this window. The user
    # PATH is read and written, not the combined one, which would copy every
    # system-wide entry into the user's own.
    function Add-ToPath([string]$Prefix) {
        $user = [Environment]::GetEnvironmentVariable('PATH', 'User')
        if (-not $user) { $user = '' }
        $parts = @($user -split ';' | Where-Object { $_ })
        if ($parts -notcontains $Prefix) {
            [Environment]::SetEnvironmentVariable('PATH', (@($parts + $Prefix) -join ';'), 'User')
            Write-Log 'Path' "added $Prefix to your PATH, so new terminals will find smallm"
        }
        if (@($env:PATH -split ';') -notcontains $Prefix) {
            $env:PATH = "$env:PATH;$Prefix"
        }
    }

    function Remove-FromPath([string]$Prefix) {
        $user = [Environment]::GetEnvironmentVariable('PATH', 'User')
        if (-not $user) { return }
        $parts = @($user -split ';' | Where-Object { $_ -and $_ -ne $Prefix })
        [Environment]::SetEnvironmentVariable('PATH', ($parts -join ';'), 'User')
    }

    function Install-Smallm {
        $repo = if ($env:SMALLM_REPO) { $env:SMALLM_REPO } else { 'https://github.com/mattsaund/SmaLLM.git' }
        $ref = if ($env:SMALLM_REF) { $env:SMALLM_REF } else { 'main' }
        $prefix = if ($env:SMALLM_PREFIX) { $env:SMALLM_PREFIX } else { Join-Path $HOME '.local\bin' }
        $binary = Join-Path $prefix 'smallm.exe'

        if ($Uninstall) {
            # The program knows what it put where, so it removes itself and
            # its models, after asking.
            $knows = (Test-Path $binary) -and ((& $binary help) -join ' ') -match 'uninstall'
            if ($knows) {
                if ($Yes) { & $binary uninstall --yes } else { & $binary uninstall }
            }
            elseif (Test-Path $binary) {
                Remove-Item -Force $binary -ErrorAction Stop
                Write-Log 'Removed' $binary
            }
            else {
                Write-Log 'Absent' "no smallm.exe in $prefix"
            }
            Remove-FromPath -Prefix $prefix
            return
        }

        # Running from inside a checkout means "build this", so a checkout
        # never downloads. Nor does naming a branch: a branch today is not
        # what any release was. A version tag names its release exactly.
        $checkout = (Test-Path 'Cargo.toml') -and
            (Select-String -Path 'Cargo.toml' -Pattern '^name = "smallm"' -Quiet)
        $released = ($ref -eq 'main') -or ($ref -match '^v[0-9]')
        $installed = $false
        if (-not $checkout -and -not $env:SMALLM_FROM_SOURCE -and $released) {
            $installed = Install-Prebuilt -Repo $repo -Ref $ref -Prefix $prefix
        }
        if (-not $installed) {
            Install-FromSource -Repo $repo -Ref $ref -Prefix $prefix -Checkout $checkout
        }
        if (-not (Test-Path $binary)) { throw "expected a binary at $binary" }

        Set-Progress 100 'installed'
        $version = (& $binary version) -join ' '
        Write-Log 'Installed' ('{0} in {1}' -f $version, $binary)
        Clear-Bar
        $screen.Live = $false
        Add-ToPath -Prefix $prefix
        # `smallm pull` downloads with curl.exe, part of Windows since 10.
        if (-not (Have 'curl.exe')) {
            Write-Warn 'Missing' "curl.exe, which 'smallm pull' downloads with. Install it from https://curl.se/windows"
        }
        Write-Log 'Next' 'get a model:   smallm pull bartowski/SmolLM2-135M-Instruct-GGUF'
        Write-Log '' 'chat with it:  smallm run SmolLM2-135M'
    }

    $screen.Work = Join-Path ([IO.Path]::GetTempPath()) ('smallm-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $screen.Work | Out-Null
    try {
        Install-Smallm
        # A clean code when it worked. The download path runs no program of
        # its own, so without this a caller reads whatever was there before.
        $global:LASTEXITCODE = 0
    }
    catch {
        Clear-Bar
        $screen.Live = $false
        Write-Warn 'Failed' $_.Exception.Message
        # Not exit, which would close the window. A failed exit code is
        # still there for anything that checks: a CI job, a script.
        $global:LASTEXITCODE = 1
    }
    finally {
        Clear-Bar
        Remove-Item -Recurse -Force $screen.Work -ErrorAction SilentlyContinue
    }
} $Uninstall.IsPresent $Yes.IsPresent
