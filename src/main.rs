use std::time::Instant;

use anyhow::Result;
use bgm::{read_rgb_tensor, to_yuyv_vec, BGModel};
use clap::Clap;
use v4l::{
    buffer::Type::{VideoCapture, VideoOutput},
    io::traits::OutputStream,
    prelude::MmapStream,
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
}

const BUF_COUNT: u32 = 4;
const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
fn main() -> Result<()> {
    let args: Args = Args::parse();
    let device = tch::Device::cuda_if_available();
    eprintln!("Device: {:?}", device);
    eprintln!("Loading model ...");
    let model = BGModel::load(&args.model_checkpoint, device)?;

    let mut input = v4l::Device::with_path(&args.source_device)?;
    let mut input_fmt = Capture::format(&input)?;
    input_fmt.width = WIDTH;
    input_fmt.height = HEIGHT;
    input_fmt.fourcc = FourCC::new(b"YUYV");
    Capture::set_format(&input, &input_fmt)?;
    let input_fmt = Capture::format(&input)?;
    eprintln!("{}", input_fmt);

    let mut output = v4l::Device::with_path(&args.target_device)?;
    Output::set_format(&output, &input_fmt)?;
    let output_fmt = Output::format(&output)?;
    eprintln!("{}", output_fmt);

    let mut input_stream = MmapStream::with_buffers(&mut input, VideoCapture, BUF_COUNT)?;
    let mut output_stream = MmapStream::with_buffers(&mut output, VideoOutput, BUF_COUNT)?;

    eprintln!("Capturing background ...");
    let background = read_rgb_tensor(&mut input_stream, WIDTH, HEIGHT)?;

    eprintln!("Started streaming");

    let mut steps = 0;
    let mut prev = Instant::now();
    loop {
        let frame = read_rgb_tensor(&mut input_stream, WIDTH, HEIGHT)?;
        let output = model.crop(frame, background.to(device))?;
        let yuyv = to_yuyv_vec(&output, WIDTH, HEIGHT);

        let (buf_out, buf_out_meta) = OutputStream::next(&mut output_stream)?;
        buf_out.copy_from_slice(&yuyv);
        buf_out_meta.field = 0;

        steps += 1;
        if steps == 60 {
            steps = 0;
            let nanos_60frame = prev.elapsed().as_nanos();
            let fps = 1e9 * 60.0 / nanos_60frame as f64;
            eprintln!("FPS: {}", fps);
            prev = Instant::now();
        }
    }
}
