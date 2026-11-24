//! A small regular expression engine for pre-tokenizer patterns.
//!
//! Each BPE family splits text into words with its own regular expressions
//! (see `pretok_table.rs`). The most common ones also have hand-written
//! scanners in `pretok.rs`, and this engine runs all the others.
//!
//! It knows what those patterns use, and no more:
//!
//! | Syntax                                  | Meaning                         |
//! |-----------------------------------------|---------------------------------|
//! | `a`, `\r`, `\n`, `\t`, `￿`, `\u{1F600}` | one character               |
//! | `.`                                     | any character but a line feed   |
//! | `[a-z\p{L}]`, `[^...]`, `[...&&[...]]`  | a class, negated, intersected   |
//! | `\s \S \d \D \w \W`                     | whitespace, digits, word chars  |
//! | `\p{L} \p{Lu} \P{N} \p{Han}`            | Unicode categories, and Han     |
//! | `x? x* x+ x{n} x{n,} x{n,m}`, then `?`  | repeats, greedy or lazy         |
//! | `(x) (?:x) (?i:x)`                      | groups, ASCII case folding      |
//! | `(?=x) (?!x)`                           | lookahead                       |
//! | `^ $ \b`                                | start, end, word boundary       |
//!
//! Matching works like Perl's. Alternatives are tried in order, and a greedy
//! repeat takes all it can, then gives back one character at a time. The
//! program runs on an explicit stack, so a long input cannot overflow the
//! call stack.

use super::unicode::{Category, category, is_han, is_letter, is_number};

/// Why a pattern could not be compiled.
pub(super) type ParseError = String;

/// A compiled expression.
pub(super) struct Regex {
    /// Program 0 is the expression. The others are its lookaheads.
    programs: Vec<Vec<Inst>>,
    classes: Vec<Class>,
    /// The characters a match can start with, when it cannot be empty.
    first: First,
}

/// One instruction of a program.
#[derive(Clone, Copy, Debug)]
enum Inst {
    /// One character equal to this one.
    Char(char),
    /// One character in a class.
    Class(u32),
    /// Continue at the first target, and on failure at the second.
    Split(u32, u32),
    Jump(u32),
    /// Zero width: a lookahead program matches here, or with `negate` does
    /// not.
    Look {
        program: u32,
        negate: bool,
    },
    /// Zero width: at the start of the text.
    Start,
    /// Zero width: at the end of the text.
    End,
    /// Zero width: between a word character and something else.
    WordBoundary,
    Match,
}

/// A parsed expression, before it is compiled.
#[derive(Clone, Debug)]
enum Node {
    Empty,
    Char(char),
    Class(u32),
    Concat(Vec<Node>),
    Alternate(Vec<Node>),
    Repeat {
        node: Box<Node>,
        min: u32,
        max: Option<u32>,
        greedy: bool,
    },
    Look {
        node: Box<Node>,
        negate: bool,
    },
    Start,
    End,
    WordBoundary,
}

/// A set of characters.
#[derive(Clone, Debug, Default)]
struct Class {
    items: Vec<Item>,
    negated: bool,
    /// Members must also be members of this class: `[a&&[b]]`.
    and: Option<Box<Class>>,
    /// Membership of the 128 ASCII characters, worked out once.
    ascii: u128,
}

/// One part of a class.
#[derive(Clone, Debug)]
enum Item {
    Range(char, char),
    /// A major category such as `L`, or with `exact`, one such as `Lu`.
    Category {
        major: char,
        exact: Option<Category>,
        negated: bool,
    },
    Han(bool),
    Space(bool),
    Digit(bool),
    Word(bool),
    Nested(Box<Class>),
}

impl Item {
    fn contains(&self, c: char) -> bool {
        match *self {
            Item::Range(low, high) => low <= c && c <= high,
            Item::Category {
                major,
                exact,
                negated,
            } => {
                let found = category(c);
                let inside = match exact {
                    Some(exact) => found == exact,
                    None => found.major() == major,
                };
                inside != negated
            }
            Item::Han(negated) => is_han(c) != negated,
            Item::Space(negated) => c.is_whitespace() != negated,
            Item::Digit(negated) => (category(c) == Category::Nd) != negated,
            Item::Word(negated) => is_word(c) != negated,
            Item::Nested(ref class) => class.contains(c),
        }
    }
}

impl Class {
    fn contains(&self, c: char) -> bool {
        if c.is_ascii() {
            return self.ascii >> (c as u32) & 1 == 1;
        }
        self.contains_slow(c)
    }

    fn contains_slow(&self, c: char) -> bool {
        let inside = self.items.iter().any(|item| item.contains(c)) != self.negated;
        inside && self.and.as_ref().is_none_or(|other| other.contains_slow(c))
    }

    /// Works out the ASCII bitmap, for this class and the ones inside it.
    fn finish(mut self) -> Class {
        for item in &mut self.items {
            if let Item::Nested(inner) = item {
                **inner = std::mem::take(&mut **inner).finish();
            }
        }
        if let Some(other) = self.and.take() {
            self.and = Some(Box::new(other.finish()));
        }
        self.ascii = (0u8..128).fold(0u128, |bits, byte| {
            bits | (u128::from(self.contains_slow(byte as char)) << byte)
        });
        self
    }
}

/// A word character for `\w` and `\b`: a letter, mark, number or `_`.
fn is_word(c: char) -> bool {
    c == '_' || is_letter(c) || is_number(c) || category(c).major() == 'M'
}

/// The characters a match can start with.
#[derive(Clone, Copy, Debug, Default)]
struct First {
    ascii: u128,
    /// Whether a character beyond ASCII may start a match.
    other: bool,
    /// Whether a match may be empty, which rules out skipping ahead.
    empty: bool,
}

impl First {
    fn union(self, other: First) -> First {
        First {
            ascii: self.ascii | other.ascii,
            other: self.other || other.other,
            empty: self.empty || other.empty,
        }
    }

    fn may_start(&self, c: char) -> bool {
        self.empty
            || if c.is_ascii() {
                self.ascii >> (c as u32) & 1 == 1
            } else {
                self.other
            }
    }
}

/// Reads a pattern into nodes.
struct Parser<'a> {
    chars: Vec<char>,
    pos: usize,
    classes: &'a mut Vec<Class>,
    /// Inside `(?i:...)`: ASCII letters match either case.
    fold: bool,
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn eat_str(&mut self, text: &str) -> bool {
        let wanted: Vec<char> = text.chars().collect();
        if self.chars[self.pos..].starts_with(&wanted) {
            self.pos += wanted.len();
            true
        } else {
            false
        }
    }

    fn error(&self, what: &str) -> ParseError {
        format!("{what} at position {}", self.pos)
    }

    fn class_node(&mut self, class: Class) -> Node {
        self.classes.push(class.finish());
        Node::Class(self.classes.len() as u32 - 1)
    }

    /// `a|b|c`
    fn alternation(&mut self) -> Result<Node, ParseError> {
        let mut branches = vec![self.sequence()?];
        while self.eat('|') {
            branches.push(self.sequence()?);
        }
        Ok(if branches.len() == 1 {
            branches.pop().unwrap_or(Node::Empty)
        } else {
            Node::Alternate(branches)
        })
    }

    /// Atoms with their repeats, up to `|`, `)` or the end.
    fn sequence(&mut self) -> Result<Node, ParseError> {
        let mut nodes = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let atom = self.atom()?;
            nodes.push(self.repeat(atom)?);
        }
        Ok(match nodes.len() {
            0 => Node::Empty,
            1 => nodes.pop().unwrap_or(Node::Empty),
            _ => Node::Concat(nodes),
        })
    }

    /// A repeat after an atom, if there is one.
    fn repeat(&mut self, atom: Node) -> Result<Node, ParseError> {
        let (min, max) = match self.peek() {
            Some('?') => (0, Some(1)),
            Some('*') => (0, None),
            Some('+') => (1, None),
            Some('{') => {
                let start = self.pos;
                self.pos += 1;
                let Some(min) = self.number() else {
                    // Not a count: a literal brace.
                    self.pos = start;
                    return Ok(atom);
                };
                let max = if self.eat(',') {
                    if self.peek() == Some('}') {
                        None
                    } else {
                        Some(self.number().ok_or_else(|| self.error("a bad count"))?)
                    }
                } else {
                    Some(min)
                };
                if !self.eat('}') || max.is_some_and(|max| max < min) {
                    return Err(self.error("a bad count"));
                }
                self.pos -= 1;
                (min, max)
            }
            _ => return Ok(atom),
        };
        self.pos += 1;
        let greedy = !self.eat('?');
        if self.peek() == Some('+') {
            return Err(self.error("possessive repeats are not supported"));
        }
        Ok(Node::Repeat {
            node: Box::new(atom),
            min,
            max,
            greedy,
        })
    }

    fn number(&mut self) -> Option<u32> {
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.pos == start {
            return None;
        }
        self.chars[start..self.pos]
            .iter()
            .collect::<String>()
            .parse()
            .ok()
    }

    fn atom(&mut self) -> Result<Node, ParseError> {
        let c = self.peek().ok_or_else(|| self.error("a missing atom"))?;
        self.pos += 1;
        match c {
            '(' => self.group(),
            '[' => {
                let class = self.class()?;
                Ok(self.class_node(class))
            }
            '.' => {
                let class = Class {
                    items: vec![Item::Range('\n', '\n')],
                    negated: true,
                    ..Class::default()
                };
                Ok(self.class_node(class))
            }
            '^' => Ok(Node::Start),
            '$' => Ok(Node::End),
            '\\' => {
                if self.eat('b') {
                    return Ok(Node::WordBoundary);
                }
                match self.escape()? {
                    Escaped::Char(c) => Ok(self.literal(c)),
                    Escaped::Item(item) => {
                        let class = Class {
                            items: vec![item],
                            ..Class::default()
                        };
                        Ok(self.class_node(class))
                    }
                }
            }
            '*' | '+' | '?' => Err(self.error("a repeat with nothing to repeat")),
            c => Ok(self.literal(c)),
        }
    }

    /// One character, folded to both cases inside `(?i:...)`.
    fn literal(&mut self, c: char) -> Node {
        if self.fold && c.is_ascii_alphabetic() {
            let class = Class {
                items: vec![
                    Item::Range(c.to_ascii_lowercase(), c.to_ascii_lowercase()),
                    Item::Range(c.to_ascii_uppercase(), c.to_ascii_uppercase()),
                ],
                ..Class::default()
            };
            self.class_node(class)
        } else {
            Node::Char(c)
        }
    }

    /// After `(`.
    fn group(&mut self) -> Result<Node, ParseError> {
        let outer_fold = self.fold;
        let node = if self.eat_str("?:") {
            self.alternation()?
        } else if self.eat_str("?i:") {
            self.fold = true;
            self.alternation()?
        } else if self.eat_str("?=") {
            Node::Look {
                node: Box::new(self.alternation()?),
                negate: false,
            }
        } else if self.eat_str("?!") {
            Node::Look {
                node: Box::new(self.alternation()?),
                negate: true,
            }
        } else if self.peek() == Some('?') {
            return Err(self.error("an unsupported group"));
        } else {
            self.alternation()?
        };
        self.fold = outer_fold;
        if !self.eat(')') {
            return Err(self.error("a missing )"));
        }
        Ok(node)
    }

    /// After `[`, up to and including `]`.
    fn class(&mut self) -> Result<Class, ParseError> {
        let mut class = Class {
            negated: self.eat('^'),
            ..Class::default()
        };
        let mut first = true;
        loop {
            let c = self.peek().ok_or_else(|| self.error("a missing ]"))?;
            if c == ']' && !first {
                self.pos += 1;
                break;
            }
            first = false;
            if self.eat_str("&&") {
                if !self.eat('[') {
                    return Err(self.error("an intersection without a class"));
                }
                let other = self.class()?;
                class.and = Some(Box::new(other));
                continue;
            }
            if c == '[' {
                self.pos += 1;
                let nested = self.class()?;
                class.items.push(Item::Nested(Box::new(nested)));
                continue;
            }
            self.pos += 1;
            let low = if c == '\\' {
                match self.escape()? {
                    Escaped::Char(c) => c,
                    Escaped::Item(item) => {
                        class.items.push(item);
                        continue;
                    }
                }
            } else {
                c
            };
            // A range, unless the dash is the last character of the class.
            let is_range =
                self.peek() == Some('-') && self.chars.get(self.pos + 1).is_some_and(|&n| n != ']');
            if is_range {
                self.pos += 1;
                let high = match self.peek() {
                    Some('\\') => {
                        self.pos += 1;
                        match self.escape()? {
                            Escaped::Char(c) => c,
                            Escaped::Item(_) => return Err(self.error("a range ending in a class")),
                        }
                    }
                    Some(c) => {
                        self.pos += 1;
                        c
                    }
                    None => return Err(self.error("a missing ]")),
                };
                if high < low {
                    return Err(self.error("a backwards range"));
                }
                class.items.push(Item::Range(low, high));
            } else {
                class.items.push(Item::Range(low, low));
                if self.fold && low.is_ascii_alphabetic() {
                    let other = if low.is_ascii_lowercase() {
                        low.to_ascii_uppercase()
                    } else {
                        low.to_ascii_lowercase()
                    };
                    class.items.push(Item::Range(other, other));
                }
            }
        }
        Ok(class)
    }

    /// After a backslash.
    fn escape(&mut self) -> Result<Escaped, ParseError> {
        let c = self.peek().ok_or_else(|| self.error("a lone backslash"))?;
        self.pos += 1;
        Ok(match c {
            'r' => Escaped::Char('\r'),
            'n' => Escaped::Char('\n'),
            't' => Escaped::Char('\t'),
            'f' => Escaped::Char('\x0c'),
            'v' => Escaped::Char('\x0b'),
            's' => Escaped::Item(Item::Space(false)),
            'S' => Escaped::Item(Item::Space(true)),
            'd' => Escaped::Item(Item::Digit(false)),
            'D' => Escaped::Item(Item::Digit(true)),
            'w' => Escaped::Item(Item::Word(false)),
            'W' => Escaped::Item(Item::Word(true)),
            'p' | 'P' => self.property(c == 'P')?,
            'u' => Escaped::Char(self.code_point()?),
            c if !c.is_ascii_alphanumeric() => Escaped::Char(c),
            c => return Err(self.error(&format!("an unknown escape \\{c}"))),
        })
    }

    /// After `\p` or `\P`: `{L}`, `{Lu}` or `{Han}`.
    fn property(&mut self, negated: bool) -> Result<Escaped, ParseError> {
        if !self.eat('{') {
            return Err(self.error("a property without {"));
        }
        let start = self.pos;
        while self.peek().is_some_and(|c| c != '}') {
            self.pos += 1;
        }
        let name: String = self.chars[start..self.pos].iter().collect();
        if !self.eat('}') {
            return Err(self.error("a property without }"));
        }
        if name == "Han" {
            return Ok(Escaped::Item(Item::Han(negated)));
        }
        let mut letters = name.chars();
        let major = letters
            .next()
            .filter(|c| "CLMNPSZ".contains(*c))
            .ok_or_else(|| self.error(&format!("an unknown property {name}")))?;
        let exact = if name.len() == 2 {
            Some(
                (0..30u8)
                    .map(|number| super::unicode_tables::BY_NUMBER[number as usize])
                    .find(|category| category.name() == name)
                    .ok_or_else(|| self.error(&format!("an unknown property {name}")))?,
            )
        } else if name.len() == 1 {
            None
        } else {
            return Err(self.error(&format!("an unknown property {name}")));
        };
        Ok(Escaped::Item(Item::Category {
            major,
            exact,
            negated,
        }))
    }

    /// After `\u`: four hex digits, or hex digits in braces.
    fn code_point(&mut self) -> Result<char, ParseError> {
        let digits: String = if self.eat('{') {
            let start = self.pos;
            while self.peek().is_some_and(|c| c != '}') {
                self.pos += 1;
            }
            let digits = self.chars[start..self.pos].iter().collect();
            if !self.eat('}') {
                return Err(self.error("a code point without }"));
            }
            digits
        } else {
            let end = (self.pos + 4).min(self.chars.len());
            let digits = self.chars[self.pos..end].iter().collect();
            self.pos = end;
            digits
        };
        u32::from_str_radix(&digits, 16)
            .ok()
            .and_then(char::from_u32)
            .ok_or_else(|| self.error("a bad code point"))
    }
}

/// What a backslash escape stands for.
enum Escaped {
    Char(char),
    Item(Item),
}

impl Node {
    /// True when the node can match without consuming a character.
    fn can_be_empty(&self) -> bool {
        match self {
            Node::Empty | Node::Look { .. } | Node::Start | Node::End | Node::WordBoundary => true,
            Node::Char(_) | Node::Class(_) => false,
            Node::Concat(nodes) => nodes.iter().all(Node::can_be_empty),
            Node::Alternate(nodes) => nodes.iter().any(Node::can_be_empty),
            Node::Repeat { node, min, .. } => *min == 0 || node.can_be_empty(),
        }
    }

    /// The characters a match of this node can start with.
    fn first(&self, classes: &[Class]) -> First {
        match self {
            Node::Empty | Node::Look { .. } | Node::Start | Node::End | Node::WordBoundary => {
                First {
                    empty: true,
                    ..First::default()
                }
            }
            Node::Char(c) => First {
                ascii: if c.is_ascii() {
                    1u128 << (*c as u32)
                } else {
                    0
                },
                other: !c.is_ascii(),
                empty: false,
            },
            Node::Class(index) => First {
                ascii: classes[*index as usize].ascii,
                other: true,
                empty: false,
            },
            Node::Concat(nodes) => {
                let mut first = First {
                    empty: true,
                    ..First::default()
                };
                for node in nodes {
                    let next = node.first(classes);
                    first = First {
                        empty: next.empty,
                        ..first.union(next)
                    };
                    if !next.empty {
                        break;
                    }
                }
                first
            }
            Node::Alternate(nodes) => nodes.iter().fold(First::default(), |first, node| {
                first.union(node.first(classes))
            }),
            Node::Repeat { node, min, .. } => {
                let inner = node.first(classes);
                First {
                    empty: *min == 0 || inner.empty,
                    ..inner
                }
            }
        }
    }
}

/// Turns nodes into programs.
struct Compiler {
    programs: Vec<Vec<Inst>>,
}

impl Compiler {
    /// Compiles `node` as a new program, and returns its number.
    fn program(&mut self, node: &Node) -> Result<u32, ParseError> {
        let index = self.programs.len();
        self.programs.push(Vec::new());
        let mut code = Vec::new();
        self.emit(node, &mut code)?;
        code.push(Inst::Match);
        self.programs[index] = code;
        Ok(index as u32)
    }

    fn emit(&mut self, node: &Node, code: &mut Vec<Inst>) -> Result<(), ParseError> {
        match node {
            Node::Empty => {}
            Node::Char(c) => code.push(Inst::Char(*c)),
            Node::Class(index) => code.push(Inst::Class(*index)),
            Node::Start => code.push(Inst::Start),
            Node::End => code.push(Inst::End),
            Node::WordBoundary => code.push(Inst::WordBoundary),
            Node::Concat(nodes) => {
                for node in nodes {
                    self.emit(node, code)?;
                }
            }
            Node::Alternate(branches) => {
                // Split(this, next) before each branch but the last, and a
                // jump past the rest after each.
                let mut jumps = Vec::new();
                for (index, branch) in branches.iter().enumerate() {
                    if index + 1 < branches.len() {
                        let split = code.len();
                        code.push(Inst::Split(0, 0));
                        self.emit(branch, code)?;
                        jumps.push(code.len());
                        code.push(Inst::Jump(0));
                        code[split] = Inst::Split(split as u32 + 1, code.len() as u32);
                    } else {
                        self.emit(branch, code)?;
                    }
                }
                let end = code.len() as u32;
                for jump in jumps {
                    code[jump] = Inst::Jump(end);
                }
            }
            Node::Repeat {
                node,
                min,
                max,
                greedy,
            } => {
                for _ in 0..*min {
                    self.emit(node, code)?;
                }
                match max {
                    None => {
                        // An empty match inside an endless repeat would loop
                        // forever. No pattern in use needs one.
                        if node.can_be_empty() {
                            return Err(
                                "an endless repeat of something that can be empty".to_string()
                            );
                        }
                        let top = code.len();
                        code.push(Inst::Split(0, 0));
                        self.emit(node, code)?;
                        code.push(Inst::Jump(top as u32));
                        let exit = code.len() as u32;
                        code[top] = if *greedy {
                            Inst::Split(top as u32 + 1, exit)
                        } else {
                            Inst::Split(exit, top as u32 + 1)
                        };
                    }
                    Some(max) => {
                        // Each optional copy may be skipped, which skips
                        // the ones after it too.
                        let mut splits = Vec::new();
                        for _ in *min..*max {
                            splits.push(code.len());
                            code.push(Inst::Split(0, 0));
                            self.emit(node, code)?;
                        }
                        let exit = code.len() as u32;
                        for split in splits {
                            code[split] = if *greedy {
                                Inst::Split(split as u32 + 1, exit)
                            } else {
                                Inst::Split(exit, split as u32 + 1)
                            };
                        }
                    }
                }
            }
            Node::Look { node, negate } => {
                let program = self.program(node)?;
                code.push(Inst::Look {
                    program,
                    negate: *negate,
                });
            }
        }
        Ok(())
    }
}

impl Regex {
    /// Compiles a pattern.
    pub fn new(pattern: &str) -> Result<Regex, ParseError> {
        let mut classes = Vec::new();
        let mut parser = Parser {
            chars: pattern.chars().collect(),
            pos: 0,
            classes: &mut classes,
            fold: false,
        };
        let node = parser.alternation()?;
        if parser.pos < parser.chars.len() {
            return Err(parser.error("an unmatched )"));
        }
        let first = node.first(&classes);
        let mut compiler = Compiler {
            programs: Vec::new(),
        };
        compiler.program(&node)?;
        Ok(Regex {
            programs: compiler.programs,
            classes,
            first,
        })
    }

    /// Where a match that starts at `start` ends, trying alternatives in
    /// order. `None` when no match starts there.
    fn match_at(
        &self,
        program: usize,
        text: &[char],
        start: usize,
        stack: &mut Vec<(u32, u32)>,
    ) -> Option<usize> {
        let code = &self.programs[program];
        let base = stack.len();
        stack.push((0, start as u32));
        let found = 'search: loop {
            let Some((mut pc, pos)) = (stack.len() > base).then(|| stack.pop()).flatten() else {
                break None;
            };
            let mut pos = pos as usize;
            loop {
                match code[pc as usize] {
                    Inst::Char(c) => {
                        if text.get(pos) != Some(&c) {
                            break;
                        }
                        pos += 1;
                        pc += 1;
                    }
                    Inst::Class(index) => {
                        match text.get(pos) {
                            Some(&c) if self.classes[index as usize].contains(c) => {}
                            _ => break,
                        }
                        pos += 1;
                        pc += 1;
                    }
                    Inst::Split(first, second) => {
                        stack.push((second, pos as u32));
                        pc = first;
                    }
                    Inst::Jump(target) => pc = target,
                    Inst::Look { program, negate } => {
                        let found = self.match_at(program as usize, text, pos, stack).is_some();
                        if found == negate {
                            break;
                        }
                        pc += 1;
                    }
                    Inst::Start => {
                        if pos != 0 {
                            break;
                        }
                        pc += 1;
                    }
                    Inst::End => {
                        if pos != text.len() {
                            break;
                        }
                        pc += 1;
                    }
                    Inst::WordBoundary => {
                        let before = pos > 0 && is_word(text[pos - 1]);
                        let after = pos < text.len() && is_word(text[pos]);
                        if before == after {
                            break;
                        }
                        pc += 1;
                    }
                    Inst::Match => break 'search Some(pos),
                }
            }
        };
        stack.truncate(base);
        found
    }

    /// Splits `text` into the matches of the expression and the stretches
    /// between them, the way the BPE pre-tokenizers do. Appends where each
    /// piece ends, offset by `base`, to `ends`.
    ///
    /// An empty match only marks a place to split.
    pub fn split(&self, text: &[char], base: usize, ends: &mut Vec<usize>) {
        let mut stack = Vec::new();
        let push = |end: usize, ends: &mut Vec<usize>| {
            if ends.last().is_none_or(|&last| last < base + end) && end > 0 {
                ends.push(base + end);
            }
        };
        let mut pos = 0;
        while pos <= text.len() {
            // Skip places no match can start.
            if pos < text.len() && !self.first.may_start(text[pos]) {
                pos += 1;
                continue;
            }
            match self.match_at(0, text, pos, &mut stack) {
                Some(end) => {
                    push(pos, ends);
                    push(end, ends);
                    pos = if end > pos { end } else { pos + 1 };
                }
                None => pos += 1,
            }
        }
        push(text.len(), ends);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pieces `pattern` splits `text` into.
    fn pieces(pattern: &str, text: &str) -> Vec<String> {
        let regex = Regex::new(pattern).unwrap_or_else(|error| panic!("{pattern}: {error}"));
        let chars: Vec<char> = text.chars().collect();
        let mut ends = Vec::new();
        regex.split(&chars, 0, &mut ends);
        let mut start = 0;
        ends.iter()
            .map(|&end| {
                let piece: String = chars[start..end].iter().collect();
                start = end;
                piece
            })
            .collect()
    }

    #[test]
    fn literals_classes_and_alternatives() {
        assert_eq!(pieces("ab|c", "xabcy"), ["x", "ab", "c", "y"]);
        assert_eq!(pieces("[a-c]+", "zabcz"), ["z", "abc", "z"]);
        assert_eq!(pieces("[^a-c]+", "zabcz"), ["z", "abc", "z"]);
        assert_eq!(pieces("\\d+", "a12b3"), ["a", "12", "b", "3"]);
        assert_eq!(pieces("\\s+", "a  b"), ["a", "  ", "b"]);
        assert_eq!(
            pieces("\\u00e9+", "caf\u{e9}\u{e9}"),
            ["caf", "\u{e9}\u{e9}"]
        );
        assert_eq!(pieces("[\\u{1F600}-\\u{1F64F}]", "a😀b"), ["a", "😀", "b"]);
        // The first alternative that matches wins, even when a later one
        // would match more.
        assert_eq!(pieces("a|ab", "ab"), ["a", "b"]);
    }

    #[test]
    fn repeats_are_greedy_and_give_back() {
        assert_eq!(pieces("a{2}", "aaaaa"), ["aa", "aa", "a"]);
        assert_eq!(pieces("a{1,3}", "aaaaa"), ["aaa", "aa"]);
        assert_eq!(pieces("a{2,}", "aaaaab"), ["aaaaa", "b"]);
        assert_eq!(pieces("a+?", "aaa"), ["a", "a", "a"]);
        // `\s+(?!\S)` gives back one space so the last one joins the word.
        assert_eq!(pieces("\\s+(?!\\S)|\\S+", "a   b"), ["a", "  ", " ", "b"]);
    }

    #[test]
    fn categories_and_properties() {
        assert_eq!(pieces("\\p{L}+", "abc123Déjà"), ["abc", "123", "Déjà"]);
        assert_eq!(pieces("\\p{Lu}", "aBc"), ["a", "B", "c"]);
        assert_eq!(pieces("\\P{L}+", "ab12cd"), ["ab", "12", "cd"]);
        assert_eq!(pieces("\\p{Han}+", "abc中文def"), ["abc", "中文", "def"]);
        assert_eq!(
            pieces("[\\p{L}&&[^\\p{Han}]]+", "ab中cd"),
            ["ab", "中", "cd"]
        );
        assert_eq!(pieces("[\\p{N}\\p{P}]+", "a1,2b"), ["a", "1,2", "b"]);
    }

    #[test]
    fn groups_lookahead_and_anchors() {
        assert_eq!(pieces("(?:ab)+", "ababa"), ["abab", "a"]);
        assert_eq!(
            pieces("(?i:'s|'t)", "it'S done't"),
            ["it", "'S", " done", "'t"]
        );
        assert_eq!(pieces("a(?=b)", "abac"), ["a", "bac"]);
        assert_eq!(pieces("a(?!b)", "abac"), ["ab", "a", "c"]);
        assert_eq!(pieces("\\s+$", "a  b  "), ["a  b", "  "]);
        assert_eq!(pieces("^a", "aaa"), ["a", "aa"]);
        assert_eq!(pieces("\\b", "ab cd"), ["ab", " ", "cd"]);
    }

    #[test]
    fn digits_split_in_threes_from_the_right() {
        let pattern = "\\p{N}{1,3}(?=(?:\\p{N}{3})*(?!\\p{N}))";
        assert_eq!(pieces(pattern, "1234567"), ["1", "234", "567"]);
        assert_eq!(pieces(pattern, "123456"), ["123", "456"]);
        assert_eq!(pieces(pattern, "x12y"), ["x", "12", "y"]);
        // The same, with empty matches as the only split points.
        assert_eq!(
            pieces("(?=(\\d{3})+(?!\\d))", "1234567"),
            ["1", "234", "567"]
        );
    }

    #[test]
    fn long_inputs_do_not_overflow() {
        let text = " ".repeat(200_000) + "x";
        assert_eq!(pieces("\\s+(?!\\S)|\\s+|\\S", &text).len(), 3);
        let word = "a".repeat(200_000);
        assert_eq!(pieces("(?:(?=[\\p{L}])[^A-Z])+", &word).len(), 1);
    }

    #[test]
    fn bad_patterns_are_errors() {
        for bad in [
            "(ab", "ab)", "[ab", "a**", "\\p{Q}", "a{3,1}", "(?<x>a)", "(?:a*)*", "\\q", "a++",
        ] {
            assert!(Regex::new(bad).is_err(), "{bad}");
        }
    }
}
