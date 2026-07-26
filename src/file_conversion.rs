#[cfg(feature = "mtx")]
use matrix_merchant::{
    reader::*, 
    writer::MatrixWriter,
    MatrixSize,
    Position,
};
#[cfg(feature = "mtx")]
use num_complex::Complex;

#[cfg(feature = "hdf5")]
use hdf5::Dataset;

use std::{
    error::Error, fs, hash::{DefaultHasher, Hash, Hasher}, io,
};
use crate::vector::vector_builders::{
    VectorBuilder,
    UninitVectorBuilder,
};
use std::path::{Path, PathBuf};
use std::mem::MaybeUninit;
use std::str::FromStr;
use std::fmt::Display;


use crate::vector::{
    VectorOps,
    vec_util_traits::Get, 
    vector_exprs::ConcreteVectorExpr,
};

pub trait AsText: Sized {
    type Error: Error;

    fn as_text(&self) -> String;
    fn as_text_utf8_bytes(&self) -> Vec<u8> {
        self.as_text().into_bytes()
    }
    fn from_text(text: &str) -> Result<Self, Self::Error>;
}

macro_rules! impl_AsText_for_Display_n_FromStr {
    ($($(#[cfg($($tt:tt)*)])? $ty:ty;)*) => {
        $(
            $(#[cfg($($tt)*)])?
            impl AsText for $ty {
                type Error = <Self as FromStr>::Err;
    
                fn as_text(&self) -> String {
                    format!("{}", self)
                }
                fn from_text(text: &str) -> Result<Self, Self::Error> {
                    text.parse::<$ty>()
                }
            }
        )*
    };
}

impl_AsText_for_Display_n_FromStr!(
    u8;
    u16;
    u32;
    u64;
    u128;
    usize;
    i8;
    i16;
    i32;
    i64;
    i128;
    isize;
    String;
    //f16;
    f32;
    f64;
);

#[cfg(feature = "mtx")]
impl<T: AsText> AsText for Complex<T> where Complex<T>: Display + FromStr, <Complex<T> as FromStr>::Err: Error {
    type Error = <Self as FromStr>::Err;
    
    fn as_text(&self) -> String {
        format!("{}", self)
    }
    fn from_text(text: &str) -> Result<Self, Self::Error> {
        text.parse::<Self>()
    }
}

pub trait AsData: Sized {
    type Error: Error;

    fn as_data(&self) -> Vec<u8>;
    fn from_data(data: &[u8]) -> Result<Self, Self::Error>;
}

pub trait AsConstSizedData: AsData {
    const DATA_SIZE: usize;
}

pub struct Infallible;

impl std::fmt::Debug for Infallible {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "An \"Infallible\" error has been raised. This shouldn't be possible and indicates a faulty implementation")
    }
}

impl Display for Infallible {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for Infallible {}

pub enum IncorrectNumBytes{
    SpecificSize{
        expected: usize,
        found: usize,
    },
    Insufficient,
    Excess,
}

impl std::fmt::Debug for IncorrectNumBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SpecificSize { expected, found } => 
                write!(f, "Expected {} bytes, found {} bytes", expected, found),
            Self::Insufficient => 
                write!(f, "Expected more bytes"),
            Self::Excess =>
                write!(f, "Expected fewer bytes than given"),
        }
    }
}

impl Display for IncorrectNumBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for IncorrectNumBytes {}

macro_rules! impl_AsData_for_nums {
    ($($ty:ty: $size:expr;)*) => {
        $(
            impl AsData for $ty {
                type Error = IncorrectNumBytes;

                fn as_data(&self) -> Vec<u8> {
                    Vec::from(self.to_be_bytes())
                }
                fn from_data(data: &[u8]) -> Result<Self, Self::Error> {
                    if data.len() != $size {
                        return Err(IncorrectNumBytes::SpecificSize { expected: $size, found: data.len() })
                    }
                    Ok(Self::from_be_bytes(data.try_into().unwrap()))
                }
            }

            impl AsConstSizedData for $ty {
                const DATA_SIZE: usize = $size;
            }
        )*
    };
}

impl_AsData_for_nums!(
    u8: 1;
    u16: 2;
    u32: 4;
    u64: 8;
    u128: 16;
    usize: std::mem::size_of::<usize>();
    i8: 1;
    i16: 2;
    i32: 4;
    i64: 8;
    i128: 16;
    isize: std::mem::size_of::<isize>();
    //f16: 2;
    f32: 4;
    f64: 8;
);

#[derive(Debug)]
#[cfg(feature = "csv")]
pub enum CSVError<FieldError: Error> {
    CSVError(csv::Error),
    CellOutOfBounds,
    FieldError(FieldError),
    FileError(io::Error),
}

impl<FieldError: Error> std::fmt::Display for CSVError<FieldError> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CSVError(e) => write!(f, "{}", e),
            Self::CellOutOfBounds => write!(f, "CellOutOfBounds"),
            Self::FieldError(e) => write!(f, "{}", e),
            Self::FileError(e) => write!(f, "{}", e),
        }
    }
}

impl<FieldError: Error> From<csv::Error> for CSVError<FieldError> {
    fn from(value: csv::Error) -> Self {
        Self::CSVError(value)
    }
}

impl<FieldError: Error> From<io::Error> for CSVError<FieldError> {
    fn from(value: io::Error) -> Self {
        Self::FileError(value)
    }
}

impl<FieldError: Error> Error for CSVError<FieldError> {}

#[derive(Debug)]
#[cfg(feature = "mtx")]
pub enum MTXError {
    NonVector,
    WrongType,
    MTXError(matrix_merchant::Error),
    FileError(io::Error),
}

impl From<matrix_merchant::Error> for MTXError {
    fn from(value: matrix_merchant::Error) -> Self {
        MTXError::MTXError(value)
    }
}

impl From<io::Error> for MTXError {
    fn from(value: io::Error) -> Self {
        MTXError::FileError(value)
    }
}

fn generate_tmp_file_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let mut temp_file = std::env::temp_dir();
    let mut hasher = DefaultHasher::new();
    path.as_ref().hash(&mut hasher);
    temp_file.push(format!("MathVectorTemp{:016X}", hasher.finish()));
    temp_file
}

struct IterInsert<I: Iterator> where I::Item: Clone {
    idx: usize,
    iter: I,
    insertion_item: Option<I::Item>,
    insertion_idx: usize,
    filler: I::Item,
}

impl<I: Iterator> IterInsert<I> where I::Item: Clone {
    fn new(iter: I, item: I::Item, idx: usize, filler: I::Item) -> Self {
        Self {
            idx: 0,
            iter,
            insertion_item: Some(item),
            insertion_idx: idx,
            filler,
        }
    }
}

impl<I: Iterator> Iterator for IterInsert<I> where I::Item: Clone {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let normal_item = self.iter.next();
        let out = if normal_item.is_none() & (self.idx < self.insertion_idx) {
            Some(self.filler.clone())
        } else if self.idx == self.insertion_idx {
            self.insertion_item.take()
        } else {
            normal_item
        };
        self.idx += 1;
        out
    }
}

#[derive(Debug)]
#[cfg(feature = "hdf5")]
pub enum HDF5Error {
    WrongSize,
    HDF5Error(hdf5::Error)
}

impl From<hdf5::Error> for HDF5Error {
    fn from(value: hdf5::Error) -> Self {
        Self::HDF5Error(value)
    }
}

pub trait IntoFileVector: ConcreteVectorExpr 
where 
    Self::Unwrapped: Get<Item = Self::Output>,
    Self::Output: Sized,
{
    #[cfg(feature = "csv")]
    fn write_csv_column<P: AsRef<Path>>(&self, path: P, row_start: usize, col: usize) -> Result<(), CSVError<<Self::Output as AsText>::Error>> 
    where 
        Self::Output: AsText,
    {
        let temp_path = generate_tmp_file_path(&path);
        let mut writer = csv::Writer::from_path(&temp_path)?;
        let mut base_records = if path.as_ref().exists() {
            Some(csv::Reader::from_path(&path)?)
        } else {
            None
        };

        let mut record = csv::StringRecord::new();
        for _ in 0..row_start {
            if let Some(base_records) = &mut base_records {
                base_records.read_record(&mut record)?; // Note: not sure if I need to clear or not
            }
            writer.write_record(&record)?;
        }

        for i in 0..self.size() {
            if let Some(base_records) = &mut base_records {
                base_records.read_record(&mut record)?;// Note: not sure if I need to clear or not
            }
            let field_txt = self[i].as_text();
            writer.write_record(IterInsert::new(record.iter(), &field_txt, col, ""))?;
        }

        if let Some(base_records) = &mut base_records {
            while base_records.read_record(&mut record)? { // Note: not sure if I need to clear or not
                writer.write_record(&record)?;
            }
        }

        writer.flush()?;
        // to make sure the files get closed
        drop(writer);
        drop(base_records);
        
        fs::rename(temp_path, path)?;
        Ok(())
    }

    #[cfg(feature = "csv")]
    fn write_csv_row<P: AsRef<Path>>(&self, path: P, row: usize, col_start: usize) -> Result<(), CSVError<<Self::Output as AsText>::Error>> 
    where 
        Self::Output: AsText,
    {
        let temp_path = generate_tmp_file_path(&path);
        let mut writer = csv::Writer::from_path(&temp_path)?;
        let mut base_records = if path.as_ref().exists() {
            Some(csv::Reader::from_path(&path)?)
        } else {
            None
        };

        let mut record = csv::StringRecord::new();
        for _ in 0..row {
            if let Some(base_records) = &mut base_records {
                base_records.read_record(&mut record)?; // Note: not sure if I need to clear or not
            }
            writer.write_record(&record)?;
        }

        if let Some(base_records) = &mut base_records {
            base_records.read_record(&mut record)?; // Note: not sure if I need to clear or not
        }
        let mut new_record = csv::StringRecord::new();
        let mut old_record = record.iter();
        for _ in 0..col_start {
            new_record.push_field(old_record.next().or(Some("")).unwrap())
        }
        for i in 0..self.size() {
            new_record.push_field(&self[i].as_text());
        }
        new_record.extend(old_record.skip(self.size()));

        if let Some(base_records) = &mut base_records {
            while base_records.read_record(&mut record)? { // Note: not sure if I need to clear or not
                writer.write_record(&record)?;
            }
        }

        writer.flush()?;
        // to make sure the files get closed
        drop(writer);
        drop(base_records);
        
        fs::rename(temp_path, path)?;
        todo!()
    }

    #[cfg(feature = "mtx")]
    fn write_mtx<P: AsRef<Path>, C: AsRef<str>>(&self, path: P, comment: Option<C>) -> Result<(), MTXError> where Self::Output: matrix_merchant::Field {
        let temp_path = generate_tmp_file_path(&path);
        let mut writer = MatrixWriter::new(fs::File::open(&temp_path)?, self.size(), 1);
        if let Some(comment) = comment {
            writer.add_comment(comment)?;
        }
        writer.write_array(|Position {row, col: _}| &self[row])?;

        fs::rename(temp_path, path)?;
        Ok(())
    }

    #[cfg(feature = "hdf5")]
    fn write_hdf5_dataset(&self, dataset: &Dataset) -> Result<(), HDF5Error> where Self::Output: hdf5::H5Type + Clone {
        Ok(dataset.as_writer().write(&(0..self.size()).into_iter().map(|x| self[x].clone()).collect::<Vec<_>>())?)
    }
}

impl<V: ConcreteVectorExpr> IntoFileVector for V 
where 
    Self::Unwrapped: Get<Item = Self::Output>,
    Self::Output: Sized,
{}


#[cfg(feature = "mtx")]
macro_rules! mtx_read_fns {
    ($($read_fn_name:ident $field_name:ident $ty:ty;)*) => {
        $(
            fn $read_fn_name<P: AsRef<Path>>(&self, path: P) -> Result<Self::Concrete<$ty>, MTXError> where Self: UninitVectorBuilder {
                let reader = MtxReader::new_reader(fs::File::open(path)?)?;
                match reader.matrix().unwrap() {
                    MatrixReader::MatrixArray(array_reader) => {
                        let mut uninit = self.new_uninit();
                        let mut num_fields_written = 0;
                        let mut error = None;
                    
                        let MatrixSize {num_rows, num_cols} = array_reader.size();
                        if (num_rows != 1) & (num_cols != 1) {
                            error = Some(MTXError::NonVector);
                        } 
                    
                        let MatrixArrayReader::$field_name(array_reader) = array_reader else {return Err(MTXError::WrongType)};
                        for column in array_reader {
                            let column = match column {
                                Ok(column) => column,
                                Err(e) => {error = Some(e.into()); break}
                            };
                            for field in column {
                                uninit[num_fields_written].write(field);
                                num_fields_written += 1;
                            }
                        }
                    
                        if let Some(error) = error {
                            unsafe {
                                for i in 0..num_fields_written {
                                    MaybeUninit::assume_init_drop(&mut uninit[i]);
                                }
                            }
                            return Err(error);
                        }
                        unsafe { Ok(Self::assume_init(uninit)) }
                    }
                    MatrixReader::MatrixCoordinate(coord_reader) => {
                        use matrix_merchant::Position;
                    
                        let MatrixSize {num_rows, num_cols} = coord_reader.size();
                        if (num_rows != 1) & (num_cols != 1) {
                            return Err(MTXError::NonVector);
                        }
                    
                        let mut vector = self.new_zeroed();
                        let MatrixCoordinateReader::$field_name(coord_reader) = coord_reader else {return Err(MTXError::WrongType)};
                        for field_data in coord_reader {
                            let (Position { row, col }, field) = field_data?;
                            if num_rows == 1 {
                                vector[col] = field;
                            } else {
                                vector[row] = field;
                            }
                        }
                    
                        Ok(vector)
                    }
                }
            }
        )*
    };
}

pub trait FromFileVectorBuilder: VectorBuilder {
    #[cfg(feature = "csv")]
    /// top left corner == (row = 0, col = 0), vector is read top down
    fn read_csv_column<T: AsText, P: AsRef<Path>>(&self, path: P, row_start: usize, col: usize) -> Result<Self::Concrete<T>, CSVError<T::Error>> where 
        Self: UninitVectorBuilder,
    {
        let mut csv = csv::Reader::from_path(path)?;
        let mut records = csv.records();
        for _ in 0..row_start {let _ = records.next().ok_or(CSVError::CellOutOfBounds)?;} // don't care if these individual rows are malformed
        let mut uninit = self.new_uninit();
        let mut num_fields_written = 0;
        let mut error = None;

        // Note: this section *must not return/crash* to avoid leaking
        for _ in 0..uninit.size() {
            let record = match records.next() {
                None => {
                    error = Some(CSVError::CellOutOfBounds);
                    break;
                }
                Some(Err(e)) => {
                    error = Some(CSVError::CSVError(e));
                    break;
                }
                Some(Ok(record)) => record
            };
            if col >= record.len() {
                error = Some(CSVError::CellOutOfBounds);
                break;
            }
            let field = match T::from_text(&record[col]) {
                Err(e) => {
                    error = Some(CSVError::FieldError(e));
                    break;
                }
                Ok(field) => field
            };
            uninit[num_fields_written].write(field);
            num_fields_written += 1;
        }

        if let Some(error) = error {
            unsafe {
                for i in 0..num_fields_written {
                    MaybeUninit::assume_init_drop(&mut uninit[i])
                }
            }
            return Err(error);
        }
        unsafe { Ok(Self::assume_init(uninit)) }
    }

    #[cfg(feature = "csv")]
    fn read_csv_row<T: AsText, P: AsRef<Path>>(&self, path: P, row: usize, col_start: usize) -> Result<Self::Concrete<T>, CSVError<T::Error>> where 
        Self: UninitVectorBuilder,
    {
        let mut csv = csv::Reader::from_path(path)?;
        let mut records = csv.records();
        for _ in 0..row {let _ = records.next().ok_or(CSVError::CellOutOfBounds)?;} // don't care if these individual rows are malformed
        let mut uninit = self.new_uninit();
        let mut num_fields_written = 0;
        let mut error = None;

        // Note: this section *must not return/crash* to avoid leaking
        match records.next() {
            None => {
                error = Some(CSVError::CellOutOfBounds);
            }
            Some(Err(e)) => {
                error = Some(CSVError::CSVError(e));
            }
            Some(Ok(record)) => {
                for field in record.into_iter().skip(col_start).take(uninit.size()) {
                    let field = match T::from_text(field) {
                        Err(e) => {
                            error = Some(CSVError::FieldError(e));
                            break;
                        }
                        Ok(field) => field
                    };
                    uninit[num_fields_written].write(field);
                    num_fields_written += 1;
                }
            }
        };

        if let Some(error) = error {
            unsafe {
                for i in 0..num_fields_written {
                    MaybeUninit::assume_init_drop(&mut uninit[i]);
                }
            }
            return Err(error);
        }
        unsafe { Ok(Self::assume_init(uninit)) }
    }

    #[cfg(feature = "mtx")]
    mtx_read_fns!(
        read_mtx_real Real f64;
        read_mtx_integer Integer i64;
        read_mtx_complex Complex Complex<f64>;
    );

    #[cfg(feature = "hdf5")]
    fn read_hdf5_dataset<T: hdf5::H5Type>(&self, dataset: &Dataset) -> Result<Self::Concrete<T>, HDF5Error> where Self: UninitVectorBuilder {
        let data = dataset.read_1d()?;
        if data.len() != self.size() {
            return Err(HDF5Error::WrongSize)
        }
        let mut uninit = self.new_uninit();
        let mut num_written = 0;
        for field in data {
            uninit[num_written].write(field);
            num_written += 1;
        }
        // assuming that this ^^^ is infallible'
        assert_eq!(num_written, self.size(), "math_vector error: hdf5 dataset size didn't match number of read fields");
        
        Ok(unsafe { Self::assume_init(uninit) })
    }
}

impl<V: VectorBuilder> FromFileVectorBuilder for V {}