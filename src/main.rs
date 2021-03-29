use std::time::Instant;

use anyhow::Result;
use bgm::{read_rgb_tensor, to_yuyv_vec, BGModel};
use clap::Clap;
use v4l::{
    buffer::Type::{VideoCapture, VideoOutput},
    io::traits::OutputStream,
    prelude::*,
    video::{Capture, Output},
    FourCC,
};

#[derive(Clap, Debug)]
struct Args {
    #[clap(short, long)]
    model_checkpoint: String,
    #[clap(short, long)]
    source_device: String,
    #[clap(short, long)]
    target_device: String,

    #[clap(short, long, default_value = "640")]
    width: u32,

    #[clap(short, long, default_value = "360")]
    height: u32,
}

fn main() -> Result<()> {
    let args: Args = Args::parse();
    let device = tch::Device::cuda_if_available();
    eprintln!("Device: {:?}", device);
    eprintln!("Loading model ...");
    let model = BGModel::load(&args.model_checkpoint, device)?;

    let mut input = v4l::Device::with_path(&args.source_device)?;
    let input_fmt = set_input_format(&input, args.width, args.height)?;

    let mut output = v4l::Device::with_path(&args.target_device)?;
    Output::set_format(&output, &input_fmt)?;
    eprintln!("Output format:\n{}", Output::format(&output)?);

    let mut input_stream = MmapStream::with_buffers(&mut input, VideoCapture, 4)?;
    let mut output_stream = MmapStream::with_buffers(&mut output, VideoOutput, 4)?;

    eprintln!("Capturing background ...");
    let background = read_rgb_tensor(&mut input_stream, args.width, args.height)?.to(device);

    eprintln!("Started streaming");
    let mut frame_count = 0;
    let mut prev = Instant::now();
    loop {
        let frame = read_rgb_tensor(&mut input_stream, args.width, args.height)?;
        let output = model.crop(frame, background.shallow_clone())?;
        let mut yuyv = to_yuyv_vec(&output, args.width, args.height);

        let (buf_out, buf_out_meta) = OutputStream::next(&mut output_stream)?;
        yuyv.resize(buf_out.len(), 0);
        buf_out.copy_from_slice(&yuyv);
        buf_out_meta.field = 0;

        frame_count += 1;
        let elapsed = prev.elapsed();
        if elapsed.as_millis() >= 1000 {
            let elapsed_nanos = elapsed.as_nanos();
            let fps = frame_count as f64 * 1e9 / elapsed_nanos as f64;
            eprintln!("FPS: {}", fps);

            frame_count = 0;
            prev = Instant::now();
        }
    }
}

fn set_input_format(input: &v4l::Device, width: u32, height: u32) -> Result<v4l::Format> {
    let mut input_fmt = Capture::format(input)?;
    input_fmt.width = width;
    input_fmt.height = height;
    input_fmt.fourcc = FourCC::new(b"YUYV");
    Capture::set_format(input, &input_fmt)?;
    let input_fmt = Capture::format(input)?;
    eprintln!("Input format:\n{}", input_fmt);
    Ok(input_fmt)
}
