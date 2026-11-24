//! A bounds-checked cursor over little-endian bytes.
//!
//! Model files come from the internet, so nothing in them is trusted. Every
//! read checks the remaining length and returns an error instead of
//! panicking or reading out of bounds.

use crate::error::{Error, Result};

/// Reads little-endian values from a byte slice, advancing as it goes.
pub(crate) struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Starts reading `data` at byte `pos`.
    pub fn new(data: &'a [u8], pos: usize) -> Reader<'a> {
        Reader { data, pos }
    }

    /// The offset of the next byte to be read.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Bytes left between the cursor and the end.
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Takes the next `n` bytes.
    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.data.len());
        match end {
            Some(end) => {
                let slice = &self.data[self.pos..end];
                self.pos = end;
                Ok(slice)
            }
            None => Err(Error::format(format!(
                "unexpected end of file at byte {} ({} more bytes needed)",
                self.pos, n
            ))),
        }
    }

    /// Takes the next `N` bytes as an array.
    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let slice = self.bytes(N)?;
        Ok(slice.try_into().expect("slice has the requested length"))
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.array::<1>()?[0])
    }

    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.array()?))
    }

    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.array()?))
    }

    /// Reads a 64-bit count that must fit in memory.
    ///
    /// Each counted item occupies at least `min_item_bytes` in the file, so a
    /// count larger than the remaining bytes allow is rejected here, before
    /// anything is allocated for it.
    pub fn count(&mut self, min_item_bytes: usize) -> Result<usize> {
        let raw = self.u64()?;
        let limit = (self.remaining() / min_item_bytes.max(1)) as u64;
        if raw > limit {
            return Err(Error::format(format!(
                "count {raw} at byte {} exceeds the file size",
                self.pos - 8
            )));
        }
        Ok(raw as usize)
    }

    /// Reads a string: a 64-bit length followed by that many bytes.
    pub fn string(&mut self) -> Result<&'a [u8]> {
        let len = self.count(1)?;
        self.bytes(len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_values_in_order() {
        let mut data = Vec::new();
        data.extend_from_slice(&7u32.to_le_bytes());
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(b"abc");
        data.extend_from_slice(&1.5f32.to_le_bytes());
        let mut r = Reader::new(&data, 0);
        assert_eq!(r.u32().unwrap(), 7);
        assert_eq!(r.string().unwrap(), b"abc");
        assert_eq!(r.f32().unwrap(), 1.5);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn running_off_the_end_is_an_error() {
        let data = [1u8, 2, 3];
        let mut r = Reader::new(&data, 0);
        assert!(r.u32().is_err());
        assert_eq!(r.pos(), 0);
        assert!(Reader::new(&data, 9).u8().is_err());
    }

    #[test]
    fn oversized_counts_are_rejected() {
        let mut data = Vec::new();
        data.extend_from_slice(&u64::MAX.to_le_bytes());
        data.extend_from_slice(b"xy");
        assert!(Reader::new(&data, 0).string().is_err());
        assert!(Reader::new(&data, 0).count(1).is_err());

        let mut data = Vec::new();
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(&[0; 8]);
        assert_eq!(Reader::new(&data, 0).count(2).unwrap(), 3);
        assert!(Reader::new(&data, 0).count(4).is_err());
    }
}
