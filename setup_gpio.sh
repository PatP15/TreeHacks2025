set -e  # Exit immediately if a command fails

echo "=== Starting GPIO permission fix ==="

# 1) Create gpio group if it doesn't exist
if ! getent group gpio >/dev/null; then
  echo "Creating 'gpio' group..."
  sudo groupadd gpio
else
  echo "'gpio' group already exists."
fi

# 2) Add current user to gpio group
CURRENT_USER="$(whoami)"
echo "Adding user '$CURRENT_USER' to group 'gpio'..."
sudo usermod -aG gpio "$CURRENT_USER"

# 3) Temporary fix: set the group ownership and permissions on /dev/gpiochip*
#    This will be overwritten at next reboot, so we also do a udev rule below.
if ls /dev/gpiochip* >/dev/null 2>&1; then
  echo "Updating group ownership and permissions for /dev/gpiochip* ..."
  sudo chown root:gpio /dev/gpiochip*
  sudo chmod 660 /dev/gpiochip*
else
  echo "Warning: No /dev/gpiochip* devices found. Are you sure you're on a Jetson or have GPIO enabled?"
fi

sudo chmod 777 /dev/video0

