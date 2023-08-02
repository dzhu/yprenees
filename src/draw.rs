//! A wrapper for using [`RgbImage`] as a draw target for [`embedded_graphics`].

use std::convert::Infallible;

use embedded_graphics::{
    draw_target::DrawTarget,
    geometry::Dimensions,
    pixelcolor::BinaryColor,
    prelude::{Point, Size},
    primitives::Rectangle,
    Pixel,
};
use image::{Rgb, RgbImage};

pub struct ImageDrawTargetWrapper<'a> {
    inner: &'a mut RgbImage,
    color: Rgb<u8>,
}

impl<'a> ImageDrawTargetWrapper<'a> {
    pub fn new(inner: &'a mut RgbImage, color: Rgb<u8>) -> Self {
        Self { inner, color }
    }
}

impl<'a> Dimensions for ImageDrawTargetWrapper<'a> {
    fn bounding_box(&self) -> Rectangle {
        Rectangle {
            top_left: Point { x: 0, y: 0 },
            size: Size {
                width: self.inner.width(),
                height: self.inner.height(),
            },
        }
    }
}

impl<'a> DrawTarget for ImageDrawTargetWrapper<'a> {
    type Color = BinaryColor;

    type Error = Infallible;

    fn draw_iter<I>(&mut self, pixels: I) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = embedded_graphics::Pixel<Self::Color>>,
    {
        for Pixel(coord, color) in pixels.into_iter() {
            if matches!(color, BinaryColor::On) && self.bounding_box().contains(coord) {
                let Point { x, y } = coord;
                self.inner.put_pixel(x as u32, y as u32, self.color);
            }
        }

        Ok(())
    }
}
