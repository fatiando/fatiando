# Sample script to install anaconda under windows
# Taken from https://github.com/astropy/ci-helpers
# Authors: Stuart Mumford
# Borrowed from: Olivier Grisel and Kyle Kastner
# License: BSD 3 clause

$MINICONDA_URL = "http://repo.continuum.io/miniconda/"

function DownloadMiniconda ($version, $platform_suffix) {
    $webclient = New-Object System.Net.WebClient
    $filename = "Miniconda-" + $version + "-Windows-" + $platform_suffix + ".exe"

    $url = $MINICONDA_URL + $filename

    $basedir = $pwd.Path + "\"
    $filepath = $basedir + $filename
    if (Test-Path $filename) {
        Write-Host "Reusing" $filepath
        return $filepath
    }

    # Download and retry up to 3 times in case of network transient errors.
    Write-Host "Downloading" $filename "from" $url
    $retry_attempts = 2
    for($i=0; $i -lt $retry_attempts; $i++){
        try {
            $webclient.DownloadFile($url, $filepath)
            break
        }
        Catch [Exception]{
            Start-Sleep 1
        }
   }
   if (Test-Path $filepath) {
       Write-Host "File saved at" $filepath
   } else {
       # Retry once to get the error message if any at the last try
       $webclient.DownloadFile($url, $filepath)
   }
   return $filepath
}

function InstallMiniconda ($miniconda_version, $architecture, $python_home) {
    Write-Host "Installing miniconda" $miniconda_version "for" $architecture "bit architecture to" $python_home
    if (Test-Path $python_home) {
        Write-Host $python_home "already exists, skipping."
        return $false
    }
    if ($architecture -eq "x86") {
        $platform_suffix = "x86"
    } else {
        $platform_suffix = "x86_64"
    }
    $filepath = DownloadMiniconda $miniconda_version $platform_suffix
    Write-Host "Installing" $filepath "to" $python_home
    $args = "/InstallationType=AllUsers /S /AddToPath=1 /RegisterPython=1 /D=" + $python_home
    Write-Host $filepath $args
    Start-Process -FilePath $filepath -ArgumentList $args -Wait -Passthru
    #Start-Sleep -s 15
    if (Test-Path $python_home) {
        Write-Host "Miniconda $miniconda_version ($architecture) installation complete"
    } else {
        Write-Host "Failed to install Python in $python_home"
        Exit 1
    }
}

# Install miniconda
InstallMiniconda $env:MINICONDA_VERSION $env:PLATFORM $env:PYTHON

# Set environment variables
$env:PATH = "${env:PYTHON};${env:PYTHON}\Scripts;" + $env:PATH

# Conda config
conda config --set always_yes true

#if (! $env:CONDA_CHANNELS) {
   #$CONDA_CHANNELS=@("astropy", "astropy-ci-extras", "openastronomy")
#} else {
   #$CONDA_CHANNELS=$env:CONDA_CHANNELS.split(" ")
#}
#foreach ($CONDA_CHANNEL in $CONDA_CHANNELS) {
   #conda config --add channels $CONDA_CHANNEL
#}

# Install the build and runtime dependencies of the project.
conda update -q conda

# Create a conda environment using the astropy bonus packages
conda create -q -n test python=$env:PYTHON_VERSION
activate test

# Set environment variables for environment (activate test doesn't seem to do the trick)
$env:PATH = "${env:PYTHON}\envs\test;${env:PYTHON}\envs\test\Scripts;${env:PYTHON}\envs\test\Library\bin;" + $env:PATH

# Check that we have the expected version of Python
python --version

# Check whether a specific version of Numpy is required
if ($env:NUMPY_VERSION) {
    if($env:NUMPY_VERSION -match "stable") {
        $NUMPY_OPTION = "numpy"
    } elseif($env:NUMPY_VERSION -match "dev") {
        $NUMPY_OPTION = "Cython pip".Split(" ")
    } else {
        $NUMPY_OPTION = "numpy=" + $env:NUMPY_VERSION
    }
} else {
    $NUMPY_OPTION = ""
}
