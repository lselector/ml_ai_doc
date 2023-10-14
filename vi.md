### Vi/Vim basic commands:

2 Modes - insertion & command (also ":" commands)

- The editor begins in command mode, where you can: move cursor, delete, copy(yank) / paste(put), save/restore, etc.
- Insertion mode begins upon entering an insertion or change command.
- [ESC] - ends Insertion mode
- Most commands execute as soon as you type them except for "colon" commands which execute when you press the return key.

File saving, reverting, quitting
```
:w  save changes to file 'fname'
:w  fname  save changes to file 'fname'
:wq  exit, saving changes (same as :x   or  :ZZ )
:q     quit (will quit if no changes made)
:q!    discard changes and quit
:e!    discard changes (revert to previous saved version)
```
Inserting text
```
i , I      insert before cursor, before line
a , A      append after cursor, after line
o , O      open new line after, line before
r , R      replace one char, many chars
```
Motion
```
h , j , k , l    left, down, up, right (also arrows)
w , W            forward next word, blank delimited word
e , E            forward end of word, of blank delimited word
b , B            backward beginning of word, of blank delimited word
( , )            sentence back, forward
{ , }           paragraph back, forward
0 , $            beginning, end of line
1G , G           beginning, end of file
nG or :n         line n
fc , Fc          forward, back to char c
H , M , L        top, middle, bottom of screen
```
Deleting text:
```
dw - deletes a word (type d followed by a motion)
dd          line
:d          line
x , X       character to right, left
D           to end of line
```
Yanking text (copying in a buffer): - type y followed by a motion.
```
y$ - yanks to the end of line.
yy         line
:y         line
```
Changing text: - The change command  is is performed by typing "c" followed by a motion. It is effectively a deletion command that leaves the editor in insert mode.
```
cw - change a word
C          to end of line
cc         line
```
Putting text:
```
p    put yanked text after position or after line
P    put before position or before line
```
Bufers: - Named buffers may be specifed before any deletion, change, yank, or put command.
```
"c - named buffer c (may be any lower case letter)
"adw - deletes a word into buffer a
"ap - put the contents of the buffer back in the page
```
Markers: - Named markers may be set on any line of a file. Any lower case letter may be a marker name. Markers may also be used as the limits for ranges.
```
mc     set marker c on this line
`c     goto marker c
'c     goto marker c first non-blank
```
<br>=================================

Example: how to copy  /  paste a block of text:
Put cursor on the first character of the block and set marker 'm':
mm
Move cursor to the position right after the last character of the block.
Yank from this position back to the marker 'm'  into a named buffer 'b':
"by`m
("b - defines a buffer,  y - yank command,  `m - moves to the marker)

Now move to some other place and put the buffer after the cursor:
"bp
<br>=================================

Example: cut /paste a block of text:
Put cursor on the first character of the block and set marker 'm':
mm
Move cursor to the position right after the last character of the block.
Yank from this position back to the marker 'm'  into a named buffer 'b':
"bd`m
("b - defines a buffer,  d - delete command,  `m - moves to the marker)

Now move to some other place and put the buffer after the cursor:
"bp

<br>=================================

Example: copy/paste in vim using visual mode:
v - mark first character of the block
move the cursor to te end
y - mark last character of the block and yank block to this point (or "d" to delete to this point)
move the cursor to some other place
gp - put the block starting on the line immediately after the cursor 
       (or use "p"or "P" to put on the next/previous line - as in vi)
vawy - copy a word
vaby - copy a ( .. ) block
vaBy - copy a { .. } block

<br>=================================

Search for Strings:
```
/string    search forward
?string    search backward
n , N      repeat search in same, reverse direction
:se ic     set ignore case for searches
:se noic   back to case sensitive searches
Shift-5    jump between matching parenthesis (or curlies or brackets)
```
Replace:
```
:s/pattern /string /flags  -  replace pattern with string
```
The search and replace function is accomplished with the :s command.
It is commonly used in combination with ranges or the :g command (below).
flags:
```
g , c     all on each line, confirm each
&         repeat last :s command
```

<br>=================================

Example: how to find/replace in all the file (2 methods):
```
:%s/from/to/g
:g/from/s//to/g
```

Regular Expressions:
```
. (dot)       any single character except newline
*             zero or more repeats
[...]         any character in set
[^ ...]       any character not in set
^ , $         beginning, end of line
\< , \>       beginning, end of word
\(: : :\)    grouping (putting into memory)
\n            contents of n th grouping (recalling from memory)
```

<br>=================================

Counts:

Nearly every command may be preceded by a number that specifies how many times it is to be performed. For example 5dw will delete 5 words and 3fe will move the cursor forward to the 3rd occurance of the letter e. Even insertions may be repeated conveniently with this method, say to insert the same line 100 times.

Ranges:

Ranges may precede most "colon" commands and cause them to be executed on a line or lines. For example :3,7d would delete lines 3-7. Ranges are commonly combined with the :s command to perform a replacement on several lines, as with
```
:.,$s/pattern/string/g to make a replacement from the current line to the end of the file.
:n ,m        lines n-m
:.           current line
:$           Last line
:?c          Marker c
:%           All lines
:g/pattern/  All matching lines
```

Files:
```
:w file      Write file (current file if no name given):
:r file      Read file after line
:n           Next file
:p           Previous file
:e file      Edit file
:e!    re-read current file (discard changes)
!!program    Replace line with program output
:r!command read in an output of shell command, for example:
:r!which perl
```

Other
```
J     join lines
.     repeat last text-changing command
u     undo last change
U     undo all changes on line
ctrl-L    refresh the window
```

<br>=================================

Examples:

How to find/replace in all the file (2 methods):
```
:%s/from/to/g
:g/from/s//to/g 
```

for example:
```
:g/<tab>/s//<space><space>/g
```

How to comment out current line and all following lines:
```
  . = current line
  $ = end of file
  .,$ = here to end
  %s = whole file
```
so sepending what you wanna do, try something like
```
  :%s/^/# / to comment
  :%s/^# // to uncomment
```

Here is how to do the same substituting the whole string
(just to demonstrate the use of memory variable \1)
```
:.,$s/\(.*\)/# \1/    to comment
:.,$s/^# \(.*\)/\1/   to uncomment
```

How to repeat insert 50 times:
```
50i-<ESC> - will repeat '-'50 times
```
Note: you can repeat not only one character, but any text you type between 'i' and pressing <ESC>

Inserting <ctrl-character> - press <ctrl-v> - then <ctrl-character>

mapping tab as  2 spaces:
```
:map!<space><ctrl-v><ctrl-v><ctrl-v><ctrl-i><space><ctrl-v><ctrl-v><ctrl-v><space><ctrl-v><ctrl-v><ctrl-v><space>
```
Explanation:

The format is:      :map! <key> <substitution>
For example:       :map! h hello

To enter a control character you have to precede it with <ctrl-v>.
To enter this <ctrl-v> itself - you should precede it with <ctrl-v> too.

-------
- www.jerrywang.net/vi/ - good vi tutorial
