# Background Matting v2 Zoom plugin in Rust

This is a working demo of
[main Real-Time High Resolution Background Matting repo](https://github.com/PeterL1n/BackgroundMattingV2) in Rust.

# Prerequisites

## v4l2loopback

This plugin requires Linux, because it relies on the
[v4l2loopback kernel module](https://github.com/umlaeute/v4l2loopback) to create and stream to virtual video devices.

1. Install v4l2loopback. On Debian/Ubuntu, the command is likely `sudo apt-get install v4l2loopback-utils`.

# Run

Before running the plugin, the virtual web camera device needs to be created.

```
sudo modprobe v4l2loopback devices=1
```

The above command should create a single virtual webcam at `/dev/video1` (the number may change), which is a stream
output for the plugin script. This webcam can now be selected by software such as Zoom, browsers, etc.

After downloading the TorchScript weights of your choice
[here](https://drive.google.com/drive/u/1/folders/1cbetlrKREitIgjnIikG1HdM4x72FtgBh), launch the pluging with a command
like the following:

```
cargo run --release -- \
    --model-checkpoint torchscript_resnet101_fp32.pth \
    --source-device=/dev/video0 \
    --target-device=/dev/video1
```

-   `model-checkpoint`: Filepath of your TorchScript
-   `source-device`: Path to your real camera
-   `target-device`: Path to your virtual camera.

Once the plugin is launched, it will capture the background image. After that, it will forward green matted images to
the virtual device.
