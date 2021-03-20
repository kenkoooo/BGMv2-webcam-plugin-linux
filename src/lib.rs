use anyhow::{anyhow, Result};
use tch::{CModule, Device, IValue, Kind, Tensor};
use v4l::{io::traits::CaptureStream, prelude::MmapStream};

const MODEL_KIND: Kind = Kind::Float;
const IMAGE_KIND: Kind = Kind::Uint8;

pub struct BGModel {
    model: CModule,
    device: Device,
}

impl BGModel {
    pub fn load<T: AsRef<std::path::Path>>(path: T, device: Device) -> Result<Self> {
        let mut model = CModule::load(path)?;
        model.to(device, MODEL_KIND, true);
        model.set_eval();
        Ok(Self { device, model })
    }

    pub fn forward(&self, source: Tensor, background: Tensor) -> Result<(Tensor, Tensor)> {
        let source = source.to(self.device);
        let background = background.to(self.device);
        let out = self
            .model
            .forward_is(&[IValue::Tensor(source), IValue::Tensor(background)])?;
        let tuple = out.to_tuple()?;
        let mut iter = tuple.into_iter();
        let alpha = iter
            .next()
            .ok_or(anyhow::anyhow!("invalid output"))?
            .to_tensor()?;
        let foreground = iter
            .next()
            .ok_or(anyhow::anyhow!("invalid output"))?
            .to_tensor()?;
        Ok((alpha, foreground))
    }

    pub fn crop(&self, source: Tensor, background: Tensor) -> Result<Tensor> {
        let (alpha, foreground) = self.forward(source, background)?;
        let target_background = Tensor::of_slice(&[120.0, 255.0, 155.0])
            .to(self.device)
            .view([1, 3, 1, 1])
            .to_kind(MODEL_KIND);
        let composite: Tensor = &alpha * &foreground * 255.0 + (1 - &alpha) * &target_background;
        Ok(composite.to_kind(IMAGE_KIND))
    }
}

trait IValueExt {
    fn to_tensor(self) -> Result<Tensor>;
    fn to_tuple(self) -> Result<Vec<IValue>>;
}

impl IValueExt for IValue {
    fn to_tensor(self) -> Result<Tensor> {
        if let IValue::Tensor(tensor) = self {
            Ok(tensor)
        } else {
            Err(anyhow!("{:?} is not tensor.", self))
        }
    }
    fn to_tuple(self) -> Result<Vec<IValue>> {
        if let IValue::Tuple(tuple) = self {
            Ok(tuple)
        } else {
            Err(anyhow!("{:?} is not tuple.", self))
        }
    }
}

fn yuyv2rgb(yuyv: &[u8]) -> Vec<u8> {
    assert_eq!(yuyv.len() % 2, 0);
    let length = yuyv.len() / 2;
    let mut rgb = vec![0; length * 3];
    for i in 0..length {
        let pos = i / 2 * 4;

        let y = if i % 2 == 0 {
            (yuyv[pos] as i64) << 8
        } else {
            (yuyv[pos + 2] as i64) << 8
        };
        let u = (yuyv[pos + 1] as i64) - 128;
        let v = (yuyv[pos + 3] as i64) - 128;

        let r = (y + (359 * v)) >> 8;
        let g = (y - (88 * u) - (183 * v)) >> 8;
        let b = (y + (454 * u)) >> 8;
        rgb[3 * i + 0] = r.min(255).max(0) as u8;
        rgb[3 * i + 1] = g.min(255).max(0) as u8;
        rgb[3 * i + 2] = b.min(255).max(0) as u8;
    }
    assert_eq!(rgb.len(), 3 * length);
    rgb
}

fn rgb2yuyv(rgb: &[u8]) -> Vec<u8> {
    assert_eq!(rgb.len() % 3, 0);
    let length = rgb.len() / 3;
    let mut yuyv = vec![0; length * 2];
    for i in 0..length {
        let r = rgb[3 * i + 0] as f64;
        let g = rgb[3 * i + 1] as f64;
        let b = rgb[3 * i + 2] as f64;
        let y = 0.257 * r + 0.504 * g + 0.098 * b + 16.0;
        let u = -0.148 * r - 0.291 * g + 0.439 * b + 128.0;
        let v = 0.439 * r - 0.368 * g - 0.071 * b + 128.0;
        if i % 2 == 0 {
            yuyv[2 * i + 0] = y.max(0.0).min(255.0) as u8;
            yuyv[2 * i + 1] = u.max(0.0).min(255.0) as u8;
        } else {
            yuyv[2 * i + 0] = y.max(0.0).min(255.0) as u8;
            yuyv[2 * i + 1] = v.max(0.0).min(255.0) as u8;
        }
    }
    yuyv
}

pub fn read_rgb_tensor(input_stream: &mut MmapStream, width: u32, height: u32) -> Result<Tensor> {
    let (yuyv, _) = CaptureStream::next(input_stream)?;
    let rgb = yuyv2rgb(&yuyv);
    let tensor = Tensor::of_slice(&rgb)
        .view([1, height as i64, width as i64, 3])
        .permute(&[0, 3, 1, 2])
        .to_kind(MODEL_KIND)
        / 255.0;
    Ok(tensor)
}

pub fn to_yuyv_vec(rgb_tensor: &Tensor, width: u32, height: u32) -> Vec<u8> {
    let mut rgb = vec![0; (height * width * 3) as usize];
    rgb_tensor
        .view([1, 3, height as i64, width as i64])
        .permute(&[0, 2, 3, 1])
        .copy_data_u8(&mut rgb, (height * width * 3) as usize);
    rgb2yuyv(&rgb)
}
