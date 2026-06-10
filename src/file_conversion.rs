use std::{
    error::Error, fs::File, io::{self, BufReader, BufWriter, Write, BufRead}
};

use crate::{ vector::{
    vec_util_traits::{Get, HasReuseBuf}, vector_initialization::UninitVectorExpr, vector_math::ConcreteVectorExpr, VectorOps
}};

fn sanitize_csv_entry(entry: &str) -> String {
    // basically: put it in double quotes and double each double quote within
    let mut sanitized = String::with_capacity(entry.len() + 2);
    sanitized.push('"');
    for char in entry.chars() {
        sanitized.push(char);
        if char == '"' {
            sanitized.push('"');
        }
    }
    sanitized.push('"');
    sanitized
}

pub trait AsText: Sized {
    type Error: Error;

    fn as_text(&self) -> String;
    fn as_text_utf8_bytes(&self) -> Vec<u8> {
        self.as_text().into_bytes()
    }
    fn from_text(text: &str) -> Result<Self, Self::Error>;
}

pub trait AsData<E: Error>: Sized {
    type Error: Error;

    fn as_data(&self) -> Vec<u8>;
    fn from_data(data: &[u8]) -> Result<Self, Self::Error>;
}

#[derive(Debug)]
pub enum FileReadError<E: Error> {
    FileError(io::Error),
    ParseError(E),
    FormattingError,
}

impl <E: Error> std::fmt::Display for FileReadError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileError(e) => write!(f, "{}", e),
            Self::ParseError(e) => write!(f, "{}", e),
            Self::FormattingError => write!(f, "FormattingError")
        }
    }
}

impl<E: Error> Error for FileReadError<E> {}

pub trait VectorFileConversions: UninitVectorExpr + ConcreteVectorExpr 
where 
    Self::Unwrapped: Get<Item = Self::Output>,
    Self::Output: Sized,
{
    
    /// inserts data corresponding to a CSV file of the vector formatted as a column
    /// *WARNING* does not intelligently format itself into a prexisting csv file, it just dumps the data whether or not thats correct
    /// *WARNING* does not sanitize the entries being written (ex: if they contain a comma or double quotes)
    fn raw_unsanitized_write_csv_column<'a, E>(&'a self, file: &mut File) -> Result<(), io::Error> 
    where 
        Self::ReferencedInner<'a>: HasReuseBuf<BoundTypes = <Self::ReferencedInner<'a> as Get>::BoundItems>,
        &'a Self::Output: AsText,
    {
        let mut file = BufWriter::new(file);
        for item in self.borrow().into_vec_iter() {
            file.write_all(&item.as_text_utf8_bytes())?;
            file.write_all(&"\n".as_bytes())?;
        }
        Ok(())
    }

    /// inserts data corresponding to a CSV file of the vector formatted as a column
    /// *WARNING* does not intelligently format itself into a prexisting csv file, it just dumps the data whether or not thats correct
    fn raw_write_csv_column<'a, E>(&'a self, file: &mut File) -> Result<(), io::Error> 
    where 
        Self::ReferencedInner<'a>: HasReuseBuf<BoundTypes = <Self::ReferencedInner<'a> as Get>::BoundItems>,
        &'a Self::Output: AsText,
    {
        let mut file = BufWriter::new(file);
        for item in self.borrow().into_vec_iter() {
            file.write_all(&sanitize_csv_entry(&item.as_text()).as_bytes())?;
            file.write_all(&"\n".as_bytes())?;
        }
        Ok(())
    }

    /// inserts data corresponding to a CSV file of the vector formatted as a row
    /// *WARNING* does not intelligently format itself into a prexisting csv file, it just dumps the data whether or not thats correct
    /// *WARNING* does not sanitize the entries being written (ex: if they contain a comma or double quotes)
    fn raw_unsanitized_write_csv_row<'a, E>(&'a self, file: &mut File) -> Result<(), io::Error> 
    where 
        Self::ReferencedInner<'a>: HasReuseBuf<BoundTypes = <Self::ReferencedInner<'a> as Get>::BoundItems>,
        &'a Self::Output: AsText,
    {
        let mut file = BufWriter::new(file);
        let mut iter = self.borrow().into_vec_iter();
        if let Some(item) = iter.next() {
            file.write_all(&item.as_text_utf8_bytes())?;
        }
        for item in iter {
            file.write_all(&",".as_bytes())?;
            file.write_all(&item.as_text_utf8_bytes())?;
        }
        Ok(())
    }

    /// inserts data corresponding to a CSV file of the vector formatted as a column
    /// *WARNING* does not intelligently format itself into a prexisting csv file, it just dumps the data whether or not thats correct
    fn raw_write_csv_row<'a, E>(&'a self, file: &mut File) -> Result<(), io::Error> 
    where 
        Self::ReferencedInner<'a>: HasReuseBuf<BoundTypes = <Self::ReferencedInner<'a> as Get>::BoundItems>,
        &'a Self::Output: AsText,
    {
        let mut file = BufWriter::new(file);
        let mut iter = self.borrow().into_vec_iter();
        if let Some(item) = iter.next() {
            file.write_all(&sanitize_csv_entry(&item.as_text()).as_bytes())?;
        }
        for item in iter {
            file.write_all(&",".as_bytes())?;
            file.write_all(&sanitize_csv_entry(&item.as_text()).as_bytes())?;
        }
        Ok(())
    }

    fn read_csv_column_with_builder<E: Error>(file: &mut File, column: usize, builder: Self::Builder) -> Result<Self, FileReadError<E>> 
    where 
        Self::Output: AsText<Error = E>,
    {
        macro_rules! add_entry {
            ($parser:tt, $uninit:ident, $num_entries_written:ident, $column:ident, $current_column:ident, $started_parsing_entry:ident, $error:ident, $entry:ident) => {
                if $column == $current_column {
                    let result = <Self::Output as AsText>::from_text(&$entry);
                    match result {
                        Ok(to_write) => {
                            if $num_entries_written < $uninit.size() {
                                unsafe { Self::init_index(&mut $uninit, $num_entries_written, to_write) };
                                $num_entries_written += 1;
                            }
                        }
                        Err(e) => {
                            $error = Err(FileReadError::ParseError(e));
                            break $parser;
                        }
                    }
                }
                #[allow(unused)] { //the one for the end of line naturally doesn't use this
                    $current_column += 1;
                }
                $started_parsing_entry = true;
            };
        }

        let file = BufReader::new(file);
        let mut uninit = Self::new_uninit(builder);
        let mut entry = String::new();
        let mut in_quotes = false;
        let mut started_parsing_entry = true;
        let mut num_entries_written = 0;
        let mut error = Ok(());

        'parser: for line in file.lines() {
            match line {
                Err(e) => {
                    error = Err(FileReadError::FileError(e));
                    break 'parser;
                }
                Ok(line) => {
                    let mut current_column = 0;
                    let mut chars = line.chars();
                    if started_parsing_entry {
                        if let Some('"') = chars.next() {
                            in_quotes = true;
                        } else {
                            in_quotes = false;
                        }
                        started_parsing_entry = false;
                    }
                    while let Some(char) = chars.next() {
                        match char {
                            '"' => {
                                if !in_quotes {
                                    error = Err(FileReadError::FormattingError);
                                    break 'parser;
                                } else {
                                    let next_char = chars.next();
                                    match next_char {
                                        Some('"') => {
                                            entry.push('"');
                                        }
                                        Some(',') | None => {
                                            add_entry!('parser, uninit, num_entries_written, column, current_column, started_parsing_entry, error, entry);   
                                        }
                                        _ => {
                                            error = Err(FileReadError::FormattingError);
                                            break 'parser;
                                        }
                                    }
                                }
                            }
                            ',' if !in_quotes => {
                                add_entry!('parser, uninit, num_entries_written, column, current_column, started_parsing_entry, error, entry);
                            }
                            char => {
                                entry.push(char)
                            }
                        }
                    }
                    if !in_quotes {
                        add_entry!('parser, uninit, num_entries_written, column, current_column, started_parsing_entry, error, entry);
                    }
                }
            }
        }
        if let Err(e) = error {
            unsafe {
                for i in 0..num_entries_written {
                    Self::drop_index(&mut uninit, i);
                }
                Self::drop_ots(&mut uninit);
            }
            return Err(e)
        } else {
            return Ok(unsafe { Self::assume_init(uninit) })
        }
    }
}