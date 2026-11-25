//! Reads GGUF model files.
//!
//! A GGUF file has three parts:
//!
//! 1. A header: magic, version, tensor count, metadata count.
//! 2. Metadata: key and value pairs that describe the model and tokenizer.
//! 3. A tensor table, then the tensor data, aligned to `general.alignment`.
//!
//! Parsing copies nothing large. Strings, arrays and tensor data stay in the
//! file mapping, and the accessors hand out slices of it.

mod mmap;
mod reader;

pub use mmap::Mapping;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::tensor::{TensorType, type_name};
use reader::Reader;

/// "GGUF" as a little-endian integer.
const MAGIC: u32 = 0x4655_4747;
/// Tensor data alignment when the file does not name one.
const DEFAULT_ALIGNMENT: usize = 32;
/// The most dimensions a tensor may have.
const MAX_DIMS: usize = 4;
/// The smallest possible metadata entry: empty key, type, one byte.
const MIN_ENTRY_BYTES: usize = 8 + 4 + 1;
/// The smallest possible tensor entry: empty name, no dims, type, offset.
const MIN_TENSOR_BYTES: usize = 8 + 4 + 4 + 8;

/// The type tag of a metadata value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    fn from_id(id: u32) -> Option<ValueType> {
        Some(match id {
            0 => ValueType::U8,
            1 => ValueType::I8,
            2 => ValueType::U16,
            3 => ValueType::I16,
            4 => ValueType::U32,
            5 => ValueType::I32,
            6 => ValueType::F32,
            7 => ValueType::Bool,
            8 => ValueType::String,
            9 => ValueType::Array,
            10 => ValueType::U64,
            11 => ValueType::I64,
            12 => ValueType::F64,
            _ => return None,
        })
    }

    /// Bytes per value, for the types with a fixed size.
    fn fixed_size(self) -> Option<usize> {
        match self {
            ValueType::U8 | ValueType::I8 | ValueType::Bool => Some(1),
            ValueType::U16 | ValueType::I16 => Some(2),
            ValueType::U32 | ValueType::I32 | ValueType::F32 => Some(4),
            ValueType::U64 | ValueType::I64 | ValueType::F64 => Some(8),
            ValueType::String | ValueType::Array => None,
        }
    }

    /// A short lowercase name for listings.
    pub fn name(self) -> &'static str {
        match self {
            ValueType::U8 => "u8",
            ValueType::I8 => "i8",
            ValueType::U16 => "u16",
            ValueType::I16 => "i16",
            ValueType::U32 => "u32",
            ValueType::I32 => "i32",
            ValueType::F32 => "f32",
            ValueType::Bool => "bool",
            ValueType::String => "str",
            ValueType::Array => "arr",
            ValueType::U64 => "u64",
            ValueType::I64 => "i64",
            ValueType::F64 => "f64",
        }
    }
}

/// A metadata value, borrowed from the file.
#[derive(Clone, Copy, Debug)]
pub enum Value<'a> {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(&'a [u8]),
    Array(Array<'a>),
}

impl<'a> Value<'a> {
    /// The value as an unsigned integer, when it is a non-negative integer.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    /// The value as a float, when it is any number.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            Value::I8(v) => Some(v as f64),
            Value::I16(v) => Some(v as f64),
            Value::I32(v) => Some(v as f64),
            Value::I64(v) => Some(v as f64),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    /// The value as a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(v) => Some(v),
            _ => None,
        }
    }

    /// The value as text, when it is a valid UTF-8 string.
    pub fn as_str(&self) -> Option<&'a str> {
        match *self {
            Value::Str(bytes) => std::str::from_utf8(bytes).ok(),
            _ => None,
        }
    }

    /// The value as an array.
    pub fn as_array(&self) -> Option<Array<'a>> {
        match *self {
            Value::Array(array) => Some(array),
            _ => None,
        }
    }
}

/// A metadata array, borrowed from the file.
#[derive(Clone, Copy, Debug)]
pub struct Array<'a> {
    ty: ValueType,
    len: usize,
    data: &'a [u8],
    offset: usize,
}

impl<'a> Array<'a> {
    /// The type of every element.
    pub fn elem_type(&self) -> ValueType {
        self.ty
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The raw payload. For string arrays this is the run of
    /// length-prefixed strings.
    pub fn raw(&self) -> &'a [u8] {
        self.data
    }

    /// Byte offset of the payload from the start of the file.
    pub fn file_offset(&self) -> usize {
        self.offset
    }

    /// Iterates a string array. Yields nothing for other element types.
    pub fn strings(&self) -> Strings<'a> {
        let left = if self.ty == ValueType::String {
            self.len
        } else {
            0
        };
        Strings {
            data: self.data,
            pos: 0,
            left,
        }
    }

    /// Element `index` of a numeric array, as a float.
    pub fn number(&self, index: usize) -> Option<f64> {
        let size = self.ty.fixed_size()?;
        if index >= self.len {
            return None;
        }
        let raw = &self.data[index * size..(index + 1) * size];
        Some(match self.ty {
            ValueType::U8 | ValueType::Bool => raw[0] as f64,
            ValueType::I8 => raw[0] as i8 as f64,
            ValueType::U16 => u16::from_le_bytes([raw[0], raw[1]]) as f64,
            ValueType::I16 => i16::from_le_bytes([raw[0], raw[1]]) as f64,
            ValueType::U32 => u32::from_le_bytes(raw.try_into().ok()?) as f64,
            ValueType::I32 => i32::from_le_bytes(raw.try_into().ok()?) as f64,
            ValueType::F32 => f32::from_le_bytes(raw.try_into().ok()?) as f64,
            ValueType::U64 => u64::from_le_bytes(raw.try_into().ok()?) as f64,
            ValueType::I64 => i64::from_le_bytes(raw.try_into().ok()?) as f64,
            ValueType::F64 => f64::from_le_bytes(raw.try_into().ok()?),
            ValueType::String | ValueType::Array => return None,
        })
    }
}

/// Iterator over the strings of a metadata array.
pub struct Strings<'a> {
    data: &'a [u8],
    pos: usize,
    left: usize,
}

impl<'a> Strings<'a> {
    /// Byte offset of the next string's text inside the array payload.
    pub fn offset(&self) -> usize {
        self.pos + 8
    }
}

impl<'a> Iterator for Strings<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        if self.left == 0 {
            return None;
        }
        // The payload was walked once at parse time, so these reads succeed.
        let mut reader = Reader::new(self.data, self.pos);
        let text = reader.string().ok()?;
        self.pos = reader.pos();
        self.left -= 1;
        Some(text)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

/// A byte range inside the file.
#[derive(Clone, Copy)]
struct Span {
    start: usize,
    len: usize,
}

/// A metadata value as stored: scalars by value, the rest by position.
#[derive(Clone, Copy)]
enum Stored {
    Scalar(Value<'static>),
    Str(Span),
    Array(ValueType, usize, Span),
}

/// One tensor in the file's tensor table.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    /// The tensor's name, for example `blk.0.attn_q.weight`.
    pub name: String,
    /// Dimensions, innermost first. A matrix is `[columns, rows]`.
    pub dims: Vec<usize>,
    /// The raw GGUF type id.
    pub type_id: u32,
    /// The element type, when this engine supports it.
    pub ty: Option<TensorType>,
    /// Byte offset of the data from the start of the file.
    pub offset: usize,
    /// Size of the data in bytes. Zero when the type is not supported.
    pub size: usize,
}

impl TensorInfo {
    /// Total number of elements.
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Length of one row: the innermost dimension.
    pub fn cols(&self) -> usize {
        self.dims.first().copied().unwrap_or(1)
    }

    /// Number of rows: the product of every dimension but the innermost.
    pub fn rows(&self) -> usize {
        self.dims.iter().skip(1).product()
    }

    /// The name of the element type, known or not.
    pub fn type_name(&self) -> &'static str {
        type_name(self.type_id)
    }
}

/// A parsed GGUF file.
pub struct Gguf {
    map: Arc<Mapping>,
    version: u32,
    entries: Vec<(Span, Stored)>,
    tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
    data_start: usize,
    alignment: usize,
}

impl Gguf {
    /// Opens and parses the file at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Gguf> {
        Gguf::parse(Arc::new(Mapping::open(path.as_ref())?))
    }

    /// Reads the whole file into memory and keeps it there. See
    /// [`Mapping::lock`].
    pub fn lock_in_memory(&self) -> std::io::Result<()> {
        self.map.lock()
    }

    /// Parses a file that is already in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Gguf> {
        Gguf::parse(Arc::new(Mapping::from_vec(bytes)))
    }

    /// Parses the header, metadata and tensor table of a mapped file.
    pub fn parse(map: Arc<Mapping>) -> Result<Gguf> {
        let data = map.as_slice();
        let mut reader = Reader::new(data, 0);

        if reader.remaining() < 4 || reader.u32()? != MAGIC {
            return Err(Error::format("not a GGUF file (bad magic)"));
        }
        let version = reader.u32()?;
        if version & 0xffff == 0 {
            return Err(Error::unsupported("big-endian GGUF files"));
        }
        if !(2..=3).contains(&version) {
            return Err(Error::unsupported(format!("GGUF version {version}")));
        }
        let tensor_count = reader.count(MIN_TENSOR_BYTES)?;
        let entry_count = reader.count(MIN_ENTRY_BYTES)?;

        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            let key_start = reader.pos() + 8;
            let key = reader.string()?;
            if std::str::from_utf8(key).is_err() {
                return Err(Error::format(format!(
                    "metadata key at byte {key_start} is not UTF-8"
                )));
            }
            let type_id = reader.u32()?;
            let ty = ValueType::from_id(type_id)
                .ok_or_else(|| Error::format(format!("unknown metadata type {type_id}")))?;
            let value = parse_value(&mut reader, ty)?;
            entries.push((
                Span {
                    start: key_start,
                    len: key.len(),
                },
                value,
            ));
        }

        let mut file = Gguf {
            map: Arc::clone(&map),
            version,
            entries,
            tensors: Vec::with_capacity(tensor_count),
            tensor_index: HashMap::with_capacity(tensor_count),
            data_start: 0,
            alignment: DEFAULT_ALIGNMENT,
        };
        if let Some(alignment) = file.get("general.alignment").and_then(|v| v.as_u64()) {
            if alignment == 0 || alignment > 1 << 20 || !alignment.is_power_of_two() {
                return Err(Error::format(format!(
                    "invalid general.alignment {alignment}"
                )));
            }
            file.alignment = alignment as usize;
        }

        // The tensor table. Offsets are relative to the data section, whose
        // start is only known once the whole table has been read.
        let mut relative_offsets = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = reader.string()?;
            let name = std::str::from_utf8(name)
                .map_err(|_| Error::format("tensor name is not UTF-8"))?
                .to_string();
            let n_dims = reader.u32()? as usize;
            if n_dims > MAX_DIMS {
                return Err(Error::format(format!(
                    "tensor {name} has {n_dims} dimensions"
                )));
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                let dim = usize::try_from(reader.u64()?)
                    .map_err(|_| Error::format(format!("tensor {name} is too large")))?;
                dims.push(dim);
            }
            let type_id = reader.u32()?;
            relative_offsets.push(reader.u64()?);
            let ty = TensorType::from_id(type_id);
            let size = match ty {
                Some(ty) => tensor_bytes(ty, &dims).ok_or_else(|| {
                    Error::format(format!("tensor {name} has an invalid shape {dims:?}"))
                })?,
                None => 0,
            };
            file.tensors.push(TensorInfo {
                name,
                dims,
                type_id,
                ty,
                offset: 0,
                size,
            });
        }

        file.data_start = align_up(reader.pos(), file.alignment)
            .ok_or_else(|| Error::format("tensor data offset overflows"))?;
        for (index, relative) in relative_offsets.into_iter().enumerate() {
            let tensor = &mut file.tensors[index];
            let offset = usize::try_from(relative)
                .ok()
                .and_then(|r| r.checked_add(file.data_start));
            let end = offset.and_then(|o| o.checked_add(tensor.size));
            match (offset, end) {
                (Some(offset), Some(end)) if end <= data.len() => tensor.offset = offset,
                _ => {
                    return Err(Error::format(format!(
                        "tensor {} extends past the end of the file",
                        tensor.name
                    )));
                }
            }
            file.tensor_index.insert(tensor.name.clone(), index);
        }
        Ok(file)
    }

    /// The GGUF format version of the file.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The file mapping the accessors borrow from.
    pub fn mapping(&self) -> &Arc<Mapping> {
        &self.map
    }

    /// Byte offset where tensor data begins.
    pub fn data_start(&self) -> usize {
        self.data_start
    }

    /// Looks up a metadata value by key.
    pub fn get(&self, key: &str) -> Option<Value<'_>> {
        let data = self.map.as_slice();
        self.entries
            .iter()
            .find(|(span, _)| &data[span.start..span.start + span.len] == key.as_bytes())
            .map(|(_, stored)| self.load(stored))
    }

    /// A metadata integer, when present and non-negative.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64()
    }

    /// A metadata integer narrowed to `usize`.
    pub fn get_usize(&self, key: &str) -> Option<usize> {
        usize::try_from(self.get_u64(key)?).ok()
    }

    /// A metadata number as f32.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key)?.as_f64().map(|v| v as f32)
    }

    /// A metadata boolean.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key)?.as_bool()
    }

    /// A metadata string.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    /// A metadata array.
    pub fn get_array(&self, key: &str) -> Option<Array<'_>> {
        self.get(key)?.as_array()
    }

    /// Every metadata entry, in file order.
    pub fn entries(&self) -> impl Iterator<Item = (&str, Value<'_>)> {
        let data = self.map.as_slice();
        self.entries.iter().map(move |(span, stored)| {
            // Keys were checked to be UTF-8 when the file was parsed.
            let key = std::str::from_utf8(&data[span.start..span.start + span.len]).unwrap_or("");
            (key, self.load(stored))
        })
    }

    /// The tensor table, in file order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Looks up a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index
            .get(name)
            .map(|&index| &self.tensors[index])
    }

    /// The bytes of a tensor. Empty when its type is not supported.
    pub fn tensor_data(&self, tensor: &TensorInfo) -> &[u8] {
        &self.map.as_slice()[tensor.offset..tensor.offset + tensor.size]
    }

    /// Turns a stored value into one that borrows from the file.
    fn load(&self, stored: &Stored) -> Value<'_> {
        let data = self.map.as_slice();
        match *stored {
            Stored::Scalar(value) => value,
            Stored::Str(span) => Value::Str(&data[span.start..span.start + span.len]),
            Stored::Array(ty, len, span) => Value::Array(Array {
                ty,
                len,
                data: &data[span.start..span.start + span.len],
                offset: span.start,
            }),
        }
    }
}

/// Parses one metadata value of type `ty` at the cursor.
fn parse_value(reader: &mut Reader<'_>, ty: ValueType) -> Result<Stored> {
    Ok(match ty {
        ValueType::U8 => Stored::Scalar(Value::U8(reader.u8()?)),
        ValueType::I8 => Stored::Scalar(Value::I8(reader.u8()? as i8)),
        ValueType::U16 => Stored::Scalar(Value::U16(reader.u16()?)),
        ValueType::I16 => Stored::Scalar(Value::I16(reader.u16()? as i16)),
        ValueType::U32 => Stored::Scalar(Value::U32(reader.u32()?)),
        ValueType::I32 => Stored::Scalar(Value::I32(reader.u32()? as i32)),
        ValueType::U64 => Stored::Scalar(Value::U64(reader.u64()?)),
        ValueType::I64 => Stored::Scalar(Value::I64(reader.u64()? as i64)),
        ValueType::F32 => Stored::Scalar(Value::F32(reader.f32()?)),
        ValueType::F64 => Stored::Scalar(Value::F64(reader.f64()?)),
        ValueType::Bool => Stored::Scalar(Value::Bool(reader.u8()? != 0)),
        ValueType::String => {
            let start = reader.pos() + 8;
            let text = reader.string()?;
            Stored::Str(Span {
                start,
                len: text.len(),
            })
        }
        ValueType::Array => {
            let elem_id = reader.u32()?;
            let elem = ValueType::from_id(elem_id)
                .ok_or_else(|| Error::format(format!("unknown array element type {elem_id}")))?;
            match elem.fixed_size() {
                Some(size) => {
                    let len = reader.count(size)?;
                    let start = reader.pos();
                    reader.bytes(len * size)?;
                    Stored::Array(
                        elem,
                        len,
                        Span {
                            start,
                            len: len * size,
                        },
                    )
                }
                None if elem == ValueType::String => {
                    let len = reader.count(8)?;
                    let start = reader.pos();
                    for _ in 0..len {
                        reader.string()?;
                    }
                    Stored::Array(
                        elem,
                        len,
                        Span {
                            start,
                            len: reader.pos() - start,
                        },
                    )
                }
                None => return Err(Error::unsupported("nested metadata arrays")),
            }
        }
    })
}

/// Bytes occupied by a tensor of type `ty` and shape `dims`, or `None` when
/// the shape is impossible for the type or the size overflows.
fn tensor_bytes(ty: TensorType, dims: &[usize]) -> Option<usize> {
    let cols = dims.first().copied().unwrap_or(1);
    let mut rows = 1usize;
    for &dim in dims.iter().skip(1) {
        rows = rows.checked_mul(dim)?;
    }
    ty.row_bytes(cols)?.checked_mul(rows)
}

/// Rounds `value` up to a multiple of `alignment`, a power of two.
fn align_up(value: usize, alignment: usize) -> Option<usize> {
    Some(value.checked_add(alignment - 1)? & !(alignment - 1))
}

#[cfg(test)]
pub(crate) mod testing {
    //! A minimal GGUF writer, for building model files inside tests.

    /// Accumulates metadata and tensors, then serializes them.
    #[derive(Default)]
    pub struct Builder {
        entries: Vec<u8>,
        n_entries: u64,
        tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
    }

    impl Builder {
        pub fn new() -> Builder {
            Builder::default()
        }

        fn key(&mut self, key: &str, type_id: u32) {
            self.entries
                .extend_from_slice(&(key.len() as u64).to_le_bytes());
            self.entries.extend_from_slice(key.as_bytes());
            self.entries.extend_from_slice(&type_id.to_le_bytes());
            self.n_entries += 1;
        }

        pub fn u32(&mut self, key: &str, value: u32) -> &mut Builder {
            self.key(key, 4);
            self.entries.extend_from_slice(&value.to_le_bytes());
            self
        }

        pub fn f32(&mut self, key: &str, value: f32) -> &mut Builder {
            self.key(key, 6);
            self.entries.extend_from_slice(&value.to_le_bytes());
            self
        }

        pub fn bool(&mut self, key: &str, value: bool) -> &mut Builder {
            self.key(key, 7);
            self.entries.push(value as u8);
            self
        }

        pub fn string(&mut self, key: &str, value: &str) -> &mut Builder {
            self.key(key, 8);
            self.entries
                .extend_from_slice(&(value.len() as u64).to_le_bytes());
            self.entries.extend_from_slice(value.as_bytes());
            self
        }

        pub fn strings(&mut self, key: &str, values: &[&[u8]]) -> &mut Builder {
            self.key(key, 9);
            self.entries.extend_from_slice(&8u32.to_le_bytes());
            self.entries
                .extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                self.entries
                    .extend_from_slice(&(value.len() as u64).to_le_bytes());
                self.entries.extend_from_slice(value);
            }
            self
        }

        pub fn f32s(&mut self, key: &str, values: &[f32]) -> &mut Builder {
            self.key(key, 9);
            self.entries.extend_from_slice(&6u32.to_le_bytes());
            self.entries
                .extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                self.entries.extend_from_slice(&value.to_le_bytes());
            }
            self
        }

        pub fn i32s(&mut self, key: &str, values: &[i32]) -> &mut Builder {
            self.key(key, 9);
            self.entries.extend_from_slice(&5u32.to_le_bytes());
            self.entries
                .extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                self.entries.extend_from_slice(&value.to_le_bytes());
            }
            self
        }

        pub fn bools(&mut self, key: &str, values: &[bool]) -> &mut Builder {
            self.key(key, 9);
            self.entries.extend_from_slice(&7u32.to_le_bytes());
            self.entries
                .extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                self.entries.push(u8::from(*value));
            }
            self
        }

        /// Adds a tensor. `dims` are innermost first, as GGUF stores them.
        pub fn tensor(
            &mut self,
            name: &str,
            dims: &[u64],
            type_id: u32,
            data: Vec<u8>,
        ) -> &mut Builder {
            self.tensors
                .push((name.to_string(), dims.to_vec(), type_id, data));
            self
        }

        /// Adds an f32 tensor from its values.
        pub fn tensor_f32(&mut self, name: &str, dims: &[u64], values: &[f32]) -> &mut Builder {
            let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            self.tensor(name, dims, 0, data)
        }

        /// Serializes everything into the bytes of a GGUF file.
        pub fn build(&self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(b"GGUF");
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            out.extend_from_slice(&self.n_entries.to_le_bytes());
            out.extend_from_slice(&self.entries);
            let mut offset = 0u64;
            for (name, dims, type_id, data) in &self.tensors {
                out.extend_from_slice(&(name.len() as u64).to_le_bytes());
                out.extend_from_slice(name.as_bytes());
                out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
                for dim in dims {
                    out.extend_from_slice(&dim.to_le_bytes());
                }
                out.extend_from_slice(&type_id.to_le_bytes());
                out.extend_from_slice(&offset.to_le_bytes());
                offset += (data.len() as u64).div_ceil(32) * 32;
            }
            out.resize(out.len().div_ceil(32) * 32, 0);
            for (_, _, _, data) in &self.tensors {
                out.extend_from_slice(data);
                out.resize(out.len().div_ceil(32) * 32, 0);
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::Builder;
    use super::*;

    fn sample() -> Vec<u8> {
        let mut builder = Builder::new();
        builder
            .string("general.architecture", "llama")
            .u32("llama.block_count", 2)
            .f32("llama.rope.freq_base", 10000.0)
            .bool("tokenizer.ggml.add_bos_token", true)
            .strings("tokenizer.ggml.tokens", &[b"<s>", b"a", b"bc"])
            .f32s("tokenizer.ggml.scores", &[0.0, -1.0, -2.5])
            .tensor_f32(
                "token_embd.weight",
                &[2, 3],
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )
            .tensor("blk.0.w", &[32, 1], 8, vec![7; 34]);
        builder.build()
    }

    #[test]
    fn reads_metadata_of_every_kind() {
        let file = Gguf::from_bytes(sample()).unwrap();
        assert_eq!(file.version(), 3);
        assert_eq!(file.get_str("general.architecture"), Some("llama"));
        assert_eq!(file.get_usize("llama.block_count"), Some(2));
        assert_eq!(file.get_f32("llama.rope.freq_base"), Some(10000.0));
        assert_eq!(file.get_bool("tokenizer.ggml.add_bos_token"), Some(true));
        assert!(file.get("missing").is_none());

        let tokens = file.get_array("tokenizer.ggml.tokens").unwrap();
        assert_eq!(tokens.len(), 3);
        let texts: Vec<&[u8]> = tokens.strings().collect();
        assert_eq!(texts, [&b"<s>"[..], b"a", b"bc"]);

        let scores = file.get_array("tokenizer.ggml.scores").unwrap();
        assert_eq!(scores.number(2), Some(-2.5));
        assert_eq!(scores.number(3), None);
        assert_eq!(file.entries().count(), 6);
    }

    #[test]
    fn reads_the_tensor_table() {
        let file = Gguf::from_bytes(sample()).unwrap();
        assert_eq!(file.tensors().len(), 2);
        let embd = file.tensor("token_embd.weight").unwrap();
        assert_eq!((embd.cols(), embd.rows()), (2, 3));
        assert_eq!(embd.ty, Some(TensorType::F32));
        assert_eq!(file.tensor_data(embd).len(), 24);
        assert_eq!(&file.tensor_data(embd)[..4], &1.0f32.to_le_bytes());
        assert_eq!(embd.offset % 32, 0);

        let w = file.tensor("blk.0.w").unwrap();
        assert_eq!(w.ty, Some(TensorType::Q8_0));
        assert_eq!(file.tensor_data(w), &[7; 34]);
    }

    #[test]
    fn rejects_files_that_are_not_gguf() {
        assert!(Gguf::from_bytes(Vec::new()).is_err());
        assert!(Gguf::from_bytes(b"GGML\x03\0\0\0".to_vec()).is_err());
        let mut wrong_version = sample();
        wrong_version[4] = 9;
        assert!(Gguf::from_bytes(wrong_version).is_err());
    }

    #[test]
    fn every_truncation_is_an_error_not_a_panic() {
        let bytes = sample();
        let file = Gguf::from_bytes(bytes.clone()).unwrap();
        let end_of_table = file.data_start();
        for len in 0..end_of_table {
            assert!(
                Gguf::from_bytes(bytes[..len].to_vec()).is_err(),
                "length {len}"
            );
        }
        // Cutting into tensor data is caught by the size check.
        assert!(Gguf::from_bytes(bytes[..bytes.len() - 40].to_vec()).is_err());
    }

    #[test]
    fn corrupted_bytes_never_panic() {
        // Flip every byte of the header and tables in turn. Parsing must
        // either succeed or fail cleanly.
        let bytes = sample();
        let end_of_table = Gguf::from_bytes(bytes.clone()).unwrap().data_start();
        for index in 0..end_of_table {
            for flip in [0x01u8, 0x80, 0xff] {
                let mut corrupt = bytes.clone();
                corrupt[index] ^= flip;
                if let Ok(file) = Gguf::from_bytes(corrupt) {
                    for tensor in file.tensors() {
                        let _ = file.tensor_data(tensor);
                    }
                    for (_, value) in file.entries() {
                        if let Value::Array(array) = value {
                            let _ = array.strings().count();
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tensor_shapes_must_fit_their_type() {
        let mut builder = Builder::new();
        builder.tensor("bad", &[33, 1], 8, vec![0; 34]);
        assert!(Gguf::from_bytes(builder.build()).is_err());

        let mut builder = Builder::new();
        builder.tensor("huge", &[u64::MAX / 2, 4], 0, vec![0; 16]);
        assert!(Gguf::from_bytes(builder.build()).is_err());
    }
}
