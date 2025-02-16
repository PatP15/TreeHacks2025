#!/bin/bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-arm64
# Exit when an error happens instead of continue.
set -e

# Default values for flags.
DEBUG_TYPE="Release"
NUM_JOBS=1
MOCO="on"
CORE_BRANCH="opensim_451"
GENERATOR="Unix Makefiles"

Help() {
    echo
    echo "This script builds and installs the last available version of OpenSim-Core in your computer."
    echo "Usage: ./scriptName [OPTION]..."
    echo "Example: ./opensim-core-build.sh -j 4 -d \"Release\""
    echo "    -d         Debug Type. Available Options:"
    echo "                   Release (Default): No debugger symbols. Optimized."
    echo "                   Debug: Debugger symbols. No optimizations (>10x slower). Library names ending with _d."
    echo "                   RelWithDefInfo: Debugger symbols. Optimized. Bigger than Release, but not slower."
    echo "                   MinSizeRel: No debugger symbols. Minimum size. Optimized."
    echo "    -j         Number of jobs to use when building libraries (>=1)."
    echo "    -s         Simple build without moco (Tropter and Casadi disabled)."
    echo "    -c         Branch for opensim-core repository."
    echo "    -n         Use the Ninja generator to build opensim-core. If not set, Unix Makefiles is used."
    echo
    exit
}

# Get flag values if any.
while getopts 'j:d:s:c:n' flag
do
    case "${flag}" in
        j) NUM_JOBS=${OPTARG};;
        d) DEBUG_TYPE=${OPTARG};;
        s) MOCO="off";;
        c) CORE_BRANCH=${OPTARG};;
        n) GENERATOR="Ninja";;
        *) Help;
    esac
done

# Check if parameters are valid.
if [[ $NUM_JOBS -lt 1 ]]
then
    Help;
fi
if [[ $DEBUG_TYPE != "Release" ]] && [[ $DEBUG_TYPE != "Debug" ]] && [[ $DEBUG_TYPE != "RelWithDebInfo" ]] && [[ $DEBUG_TYPE != "MinSizeRel" ]]
then
    Help;
fi

# Show values of flags:
echo
echo "Build script parameters:"
echo "DEBUG_TYPE="$DEBUG_TYPE
echo "NUM_JOBS="$NUM_JOBS
echo "MOCO="$MOCO
echo "CORE_BRANCH="$CORE_BRANCH
echo "GENERATOR="$GENERATOR
echo ""

# Check OS.
echo "LOG: CHECKING OS..."
OS_NAME=$(lsb_release -a)

if [[ $OS_NAME == *"Debian"* ]]; then
   OS_NAME="Debian"
   echo "The OS of this machine is Debian."
elif [[ $OS_NAME == *"Ubuntu"* ]]; then
   OS_NAME="Ubuntu"
   echo "The OS of this machine is Ubuntu."
else
   OS_NAME="Unknown"
   echo "Could not recognize the OS in your machine."
   exit
fi
echo


# Build opensim-core.
echo "LOG: BUILDING OPENSIM-CORE..."
mkdir -p ~/opensim-workspace/opensim-core-build || true
cd ~/opensim-workspace/opensim-core-build
cmake ~/opensim-workspace/opensim-core-source -G"$GENERATOR" \
  -DOPENSIM_DEPENDENCIES_DIR=~/opensim-workspace/opensim-core-dependencies-install/ \
  -DBUILD_JAVA_WRAPPING=on \
  -DBUILD_PYTHON_WRAPPING=on \
  -DOPENSIM_C3D_PARSER=ezc3d \
  -DBUILD_TESTING=off \
  -DCMAKE_INSTALL_PREFIX=~/opensim-core \
  -DOPENSIM_INSTALL_UNIX_FHS=off \
  -DSWIG_DIR=~/swig/share/swig \
  -DSWIG_EXECUTABLE=~/swig/bin/swig \
  -DPython3_ROOT_DIR=$(python3 -c "import sys; print(sys.prefix)") \
  -DPython3_EXECUTABLE=$(which python3) \
  -DPython3_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \
  -DPython3_LIBRARY=$(python3-config --ldflags | awk '{print $1}') \
  -DOPENSIM_PYTHON_VERSION=3 \
  -DBUILD_PYTHON_WRAPPING=ON \
  -DJAVA_HOME=$JAVA_HOME \
  -DJAVA_AWT_LIBRARY="$JAVA_HOME/lib/libjawt.so" \
  -DJAVA_JVM_LIBRARY="$JAVA_HOME/lib/server/libjvm.so" \
  -DJAVA_INCLUDE_PATH="$JAVA_HOME/include" \
  -DJAVA_INCLUDE_PATH2="$JAVA_HOME/include/linux" \
  -DJAVA_AWT_INCLUDE_PATH="$JAVA_HOME/include" \
cmake . -LAH
cmake --build . --config $DEBUG_TYPE -j$NUM_JOBS
echo

# Test opensim-core.
echo "LOG: TESTING OPENSIM-CORE..."
cd ~/opensim-workspace/opensim-core-build
# TODO: Temporary for python to find Simbody libraries.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/opensim-workspace/opensim-core-dependencies-install/simbody/lib
ctest --parallel $NUM_JOBS --output-on-failure

# Install opensim-core.
echo "LOG: INSTALL OPENSIM-CORE..."
cd ~/opensim-workspace/opensim-core-build
cmake --install .
echo 'export PATH=~/opensim-core/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
