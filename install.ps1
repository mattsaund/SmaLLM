# Install SmaLLM on Windows.
#
#   irm https://raw.githubusercontent.com/mattsaund/SmaLLM/main/install.ps1 | iex
#
# Downloads the ready-made smallm.exe from the latest GitHub release and
# checks it against the release's SHA-256 list. When there is none to
# download, it builds from source, which needs Rust and the Visual Studio
# Build Tools.
#
# To remove SmaLLM again:
#
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/mattsaund/SmaLLM/main/install.ps1))) -Uninstall
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

param([switch]$Uninstall)

& {
    param([bool]$Uninstall)
    $ErrorActionPreference = 'Continue'

    # GitHub only speaks TLS 1.2 and up, and Windows PowerShell 5.1 on an
    # older Windows does not offer it unless asked.
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    function Have([string]$Name) {
        [bool](Get-Command $Name -ErrorAction SilentlyContinue)
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

    # The release's zip for this machine, checked and unpacked into $Prefix.
    # Returns whether it worked. A missing release is a reason to build
    # instead. A checksum that does not match is a reason to stop.
    function Install-Prebuilt([string]$Repo, [string]$Ref, [string]$Prefix) {
        $slug = $Repo -replace '^https://github\.com/', '' -replace '\.git$', ''
        if ($slug -match '://' -or $slug -notmatch '^[^/]+/[^/]+$') { return $false }
        $release = if ($Ref -eq 'main') { 'latest/download' } else { "download/$Ref" }
        $base = "https://github.com/$slug/releases/$release"
        $asset = if (Test-Arm64) { 'smallm-aarch64-pc-windows-msvc.zip' } else { 'smallm-x86_64-pc-windows-msvc.zip' }
        $work = Join-Path ([IO.Path]::GetTempPath()) ('smallm-' + [Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Force -Path $work | Out-Null
        try {
            Write-Host 'downloading smallm ... ' -NoNewline
            # The progress bar of Invoke-WebRequest in 5.1 slows a download
            # down by an order of magnitude.
            $ProgressPreference = 'SilentlyContinue'
            $zip = Join-Path $work $asset
            try {
                Invoke-WebRequest -UseBasicParsing -ErrorAction Stop -OutFile $zip -Uri "$base/$asset"
            }
            catch {
                Write-Host 'none published for this machine, so building from source instead'
                return $false
            }
            $sums = Join-Path $work 'SHA256SUMS'
            try {
                Invoke-WebRequest -UseBasicParsing -ErrorAction Stop -OutFile $sums -Uri "$base/SHA256SUMS"
                $actual = (Get-FileHash -Algorithm SHA256 -Path $zip).Hash.ToLower()
                $line = Get-Content $sums | Where-Object { $_ -match ([regex]::Escape($asset) + '$') } | Select-Object -First 1
                $expected = if ($line) { ($line -split '\s+')[0].ToLower() } else { '' }
                if ($expected -ne $actual) {
                    throw 'the download does not match its checksum. Nothing was installed.'
                }
            }
            catch [System.Net.WebException] {
                Write-Host '(checksum not verified) ' -NoNewline
            }
            Expand-Archive -Path $zip -DestinationPath $work -Force -ErrorAction Stop
            $exe = Get-ChildItem -Path $work -Filter 'smallm.exe' -Recurse | Select-Object -First 1
            if (-not $exe) {
                Write-Host 'the download had no smallm.exe in it, so building instead'
                return $false
            }
            Install-Exe -From $exe.FullName -Prefix $Prefix
            Write-Host 'done'
            return $true
        }
        finally {
            Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
        }
    }

    # Build with cargo from a checkout or a fresh clone.
    function Install-FromSource([string]$Repo, [string]$Ref, [string]$Prefix, [bool]$Checkout) {
        if (-not (Have 'cargo')) {
            $cargoBin = Join-Path $HOME '.cargo\bin'
            if (Test-Path (Join-Path $cargoBin 'cargo.exe')) {
                $env:PATH = "$cargoBin;$env:PATH"
            }
            else {
                throw 'building from source needs Rust. Install it from https://rustup.rs and run this again'
            }
        }
        $source = $null
        $temp = $false
        try {
            if ($Checkout) {
                $source = (Get-Location).Path
                Write-Host "building from $source"
            }
            else {
                if (-not (Have 'git')) { throw 'git is needed to fetch the source' }
                $source = Join-Path ([IO.Path]::GetTempPath()) ('smallm-' + [Guid]::NewGuid().ToString('N'))
                $temp = $true
                Write-Host "fetching $Repo ($Ref) ... " -NoNewline
                & git clone --quiet --depth 1 --branch $Ref $Repo $source 2>$null | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    Write-Host ''
                    throw "could not clone $Repo ($Ref)"
                }
                Write-Host 'done'
            }
            Write-Host 'building (about half a minute)'
            & cargo build --release --locked --manifest-path (Join-Path $source 'Cargo.toml') --bin smallm
            if ($LASTEXITCODE -ne 0) { throw 'the build failed. The output above says why' }
            $exe = Join-Path $source 'target\release\smallm.exe'
            if (-not (Test-Path $exe)) { throw 'the build finished without leaving a smallm.exe' }
            Install-Exe -From $exe -Prefix $Prefix
        }
        finally {
            if ($temp -and $source -and (Test-Path $source)) {
                Remove-Item -Recurse -Force $source -ErrorAction SilentlyContinue
            }
        }
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
            Write-Host "added $Prefix to your PATH, so new terminals will find smallm"
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
            if (Test-Path $binary) {
                Remove-Item -Force $binary -ErrorAction Stop
                Write-Host "removed $binary"
            }
            else {
                Write-Host "no smallm.exe in $prefix"
            }
            Remove-FromPath -Prefix $prefix
            Write-Host 'Downloaded models are kept, in %LOCALAPPDATA%\smallm. Delete that folder to remove them.'
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

        Add-ToPath -Prefix $prefix
        Write-Host ''
        Write-Host "installed $binary"
        Write-Host 'get a model:   smallm pull bartowski/SmolLM2-135M-Instruct-GGUF'
        Write-Host 'chat with it:  smallm run SmolLM2-135M'
    }

    try {
        Install-Smallm
        # A clean code when it worked. The download path runs no program of
        # its own, so without this a caller reads whatever was there before.
        $global:LASTEXITCODE = 0
    }
    catch {
        Write-Host ''
        Write-Host "install: $($_.Exception.Message)" -ForegroundColor Red
        # Not exit, which would close the window. A failed exit code is
        # still there for anything that checks: a CI job, a script.
        $global:LASTEXITCODE = 1
    }
} $Uninstall.IsPresent
