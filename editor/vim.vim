syntax clear

syntax match mylangComment /\\.*$/

syntax region mylangString start=/"/ skip=/\\./ end=/"/ contains=mylangEscape
syntax match mylangEscape /\\./ contained

syntax match mylangNumber /\v0b[01]+/
syntax match mylangNumber /\v0o[0-7]+/
syntax match mylangNumber /\v0x[\da-fA-F]+/
syntax match mylangNumber /\v\d+\.\d+([eE][+-]?\d+)?/
syntax match mylangNumber /\v\d+[eE][+-]?\d+/
syntax match mylangNumber /\v\d+/

syntax keyword mylangKeyword Import Print If While

syntax keyword mylangType Real Bool Unit Type forall

syntax match mylangSpecial /--/
syntax match mylangSpecial /(/
syntax match mylangSpecial /)/
syntax match mylangSpecial /\[/
syntax match mylangSpecial /\]/
syntax match mylangSpecial /{/
syntax match mylangSpecial /}/

syntax keyword mylangBuiltin drop nip dup over swap rotr rotl apply dip

syntax match mylangVariable /!\v[a-zA-Z_][a-zA-Z0-9!'\-?_]*/
syntax match mylangFunction /@\v[a-zA-Z_][a-zA-Z0-9!'\-?_]*/

hi def link mylangComment Comment
hi def link mylangString String
hi def link mylangEscape SpecialChar
hi def link mylangNumber Number
hi def link mylangKeyword Keyword
hi def link mylangType Type
hi def link mylangSpecial Special

hi def link mylangBuiltin Function

hi def link mylangVariable Identifier
hi def link mylangFunction Function

