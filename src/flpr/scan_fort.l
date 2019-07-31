/*
   Copyright (c) 2019, Triad National Security, LLC. All rights reserved.

   This is open source software; you can redistribute it and/or modify it
   under the terms of the BSD-3 License. If software is modified to produce
   derivative works, such modified software should be clearly marked, so as
   not to confuse it with the version available from LANL. Full text of the
   BSD-3 License can be found in the LICENSE file of the repository.
*/

/* flex scanner Fortran tokens and keywords. */

%{
#include "flpr/Syntax_Tags.hh"
int curr_line_pos = 0;
#define YY_USER_ACTION curr_line_pos += yyleng;
using FLPR::Syntax_Tags;
%}
%option full
%option case-insensitive
%option noyywrap

DIGIT  [0-9]
NAME   [a-z][a-z0-9_]*
KIND   ([0-9]+)|({NAME})
SIGNIFICAND  ([0-9]+\.[0-9]*)|(\.[0-9]+)
EXPONENT     [ed][-+]?{DIGIT}+
%%
  /* Keywords */
abstract { return Syntax_Tags::KW_ABSTRACT; }
all { return Syntax_Tags::KW_ALL; }
allocatable { return Syntax_Tags::KW_ALLOCATABLE; }
allocate { return Syntax_Tags::KW_ALLOCATE; }
assignment { return Syntax_Tags::KW_ASSIGNMENT; }
associate { return Syntax_Tags::KW_ASSOCIATE; }
asynchronous { return Syntax_Tags::KW_ASYNCHRONOUS; }
backspace { return Syntax_Tags::KW_BACKSPACE; }
bind { return Syntax_Tags::KW_BIND; }
block { return Syntax_Tags::KW_BLOCK; }
call { return Syntax_Tags::KW_CALL; }
case { return Syntax_Tags::KW_CASE; }
character { return Syntax_Tags::KW_CHARACTER; }
class { return Syntax_Tags::KW_CLASS; }
close { return Syntax_Tags::KW_CLOSE; }
codimension { return Syntax_Tags::KW_CODIMENSION; }
common { return Syntax_Tags::KW_COMMON; }
complex { return Syntax_Tags::KW_COMPLEX; }
concurrent { return Syntax_Tags::KW_CONCURRENT; }
contains { return Syntax_Tags::KW_CONTAINS; }
contiguous { return Syntax_Tags::KW_CONTIGUOUS; }
continue { return Syntax_Tags::KW_CONTINUE; }
critical { return Syntax_Tags::KW_CRITICAL; }
cycle { return Syntax_Tags::KW_CYCLE; }
data { return Syntax_Tags::KW_DATA; }
deallocate { return Syntax_Tags::KW_DEALLOCATE; }
default { return Syntax_Tags::KW_DEFAULT; }
deferred { return Syntax_Tags::KW_DEFERRED; }
dimension { return Syntax_Tags::KW_DIMENSION; }
do { return Syntax_Tags::KW_DO; }
dowhile { return Syntax_Tags::KW_DOWHILE; }
double { return Syntax_Tags::KW_DOUBLE; }
elemental { return Syntax_Tags::KW_ELEMENTAL; }
else { return Syntax_Tags::KW_ELSE; }
end { return Syntax_Tags::KW_END; }
entry { return Syntax_Tags::KW_ENTRY; }
enum { return Syntax_Tags::KW_ENUM; }
enumerator { return Syntax_Tags::KW_ENUMERATOR; }
equivalence { return Syntax_Tags::KW_EQUIVALENCE; }
errmsg { return Syntax_Tags::KW_ERRMSG; }
error { return Syntax_Tags::KW_ERROR; }
exit { return Syntax_Tags::KW_EXIT; }
extends { return Syntax_Tags::KW_EXTENDS; }
external { return Syntax_Tags::KW_EXTERNAL; }
file { return Syntax_Tags::KW_FILE; }
final { return Syntax_Tags::KW_FINAL; }
flush { return Syntax_Tags::KW_FLUSH; }
forall { return Syntax_Tags::KW_FORALL; }
format { return Syntax_Tags::KW_FORMAT; }
formatted { return Syntax_Tags::KW_FORMATTED; }
function { return Syntax_Tags::KW_FUNCTION; }
generic { return Syntax_Tags::KW_GENERIC; }
go  { return Syntax_Tags::KW_GO; }
if { return Syntax_Tags::KW_IF; }
implicit { return Syntax_Tags::KW_IMPLICIT; }
import { return Syntax_Tags::KW_IMPORT; }
impure { return Syntax_Tags::KW_IMPURE; }
in { return Syntax_Tags::KW_IN; }
include { return Syntax_Tags::KW_INCLUDE; }
inout { return Syntax_Tags::KW_INOUT; }
inquire { return Syntax_Tags::KW_INQUIRE; }
integer { return Syntax_Tags::KW_INTEGER; }
intent { return Syntax_Tags::KW_INTENT; }
interface { return Syntax_Tags::KW_INTERFACE; }
intrinsic { return Syntax_Tags::KW_INTRINSIC; }
is { return Syntax_Tags::KW_IS; }
kind { return Syntax_Tags::KW_KIND; }
len { return Syntax_Tags::KW_LEN; }
local { return Syntax_Tags::KW_LOCAL; }
local_init { return Syntax_Tags::KW_LOCAL_INIT; }
logical { return Syntax_Tags::KW_LOGICAL; }
module { return Syntax_Tags::KW_MODULE; }
mold { return Syntax_Tags::KW_MOLD; }
name { return Syntax_Tags::KW_NAME; }
namelist { return Syntax_Tags::KW_NAMELIST; }
nopass { return Syntax_Tags::KW_NOPASS; }
non_intrinsic { return Syntax_Tags::KW_NON_INTRINSIC; }
non_overridable { return Syntax_Tags::KW_NON_OVERRIDABLE; }
non_recursive { return Syntax_Tags::KW_NON_RECURSIVE; }
none { return Syntax_Tags::KW_NONE; }
nullify { return Syntax_Tags::KW_NULLIFY; }
only { return Syntax_Tags::KW_ONLY; }
open { return Syntax_Tags::KW_OPEN; }
operator { return Syntax_Tags::KW_OPERATOR; }
optional { return Syntax_Tags::KW_OPTIONAL; }
out { return Syntax_Tags::KW_OUT; }
parameter { return Syntax_Tags::KW_PARAMETER; }
pass { return Syntax_Tags::KW_PASS; }
pointer { return Syntax_Tags::KW_POINTER; }
precision { return Syntax_Tags::KW_PRECISION; }
print { return Syntax_Tags::KW_PRINT; }
private { return Syntax_Tags::KW_PRIVATE; }
procedure { return Syntax_Tags::KW_PROCEDURE; }
program { return Syntax_Tags::KW_PROGRAM; }
protected { return Syntax_Tags::KW_PROTECTED; }
public { return Syntax_Tags::KW_PUBLIC; }
pure { return Syntax_Tags::KW_PURE; }
quiet { return Syntax_Tags::KW_QUIET; }
rank { return Syntax_Tags::KW_RANK; }
read { return Syntax_Tags::KW_READ; }
real {return Syntax_Tags::KW_REAL; }
recursive { return Syntax_Tags::KW_RECURSIVE; }
result { return Syntax_Tags::KW_RESULT; }
return { return Syntax_Tags::KW_RETURN; }
rewind { return Syntax_Tags::KW_REWIND; }
save { return Syntax_Tags::KW_SAVE; }
select { return Syntax_Tags::KW_SELECT; }
sequence { return Syntax_Tags::KW_SEQUENCE; }
shared { return Syntax_Tags::KW_SHARED; }
source { return Syntax_Tags::KW_SOURCE; }
stat { return Syntax_Tags::KW_STAT; }
stop { return Syntax_Tags::KW_STOP; }
submodule { return Syntax_Tags::KW_SUBMODULE; }
subroutine { return Syntax_Tags::KW_SUBROUTINE; }
target { return Syntax_Tags::KW_TARGET; }
team { return Syntax_Tags::KW_TEAM; }
team_number { return Syntax_Tags::KW_TEAM_NUMBER; }
then { return Syntax_Tags::KW_THEN; }
to { return Syntax_Tags::KW_TO; }
type { return Syntax_Tags::KW_TYPE; }
unformatted { return Syntax_Tags::KW_UNFORMATTED; }
use { return Syntax_Tags::KW_USE; }
value { return Syntax_Tags::KW_VALUE; }
volatile { return Syntax_Tags::KW_VOLATILE; }
where { return Syntax_Tags::KW_WHERE; }
while { return Syntax_Tags::KW_WHILE; }
write { return Syntax_Tags::KW_WRITE; }

  /* Symbols */
"("  { return Syntax_Tags::TK_PARENL; }
")"  { return Syntax_Tags::TK_PARENR; }
"["  { return Syntax_Tags::TK_BRACKETL; }
"]"  { return Syntax_Tags::TK_BRACKETR; }
"+"  { return Syntax_Tags::TK_PLUS; }
"-"  { return Syntax_Tags::TK_MINUS; }
"="  { return Syntax_Tags::TK_EQUAL; }
".." { return Syntax_Tags::TK_DBL_DOT; }
"::" { return Syntax_Tags::TK_DBL_COLON; }
":"  { return Syntax_Tags::TK_COLON; }
";"  { return Syntax_Tags::TK_SEMICOLON; }
"/"  { return Syntax_Tags::TK_SLASHF; }
"%"  { return Syntax_Tags::TK_PERCENT; }
","  { return Syntax_Tags::TK_COMMA; }
"**" { return Syntax_Tags::TK_POWER_OP; }
"*"  { return Syntax_Tags::TK_ASTERISK; }
"|"  { return Syntax_Tags::TK_VBAR; }
"//" { return Syntax_Tags::TK_CONCAT; }
"=>" { return Syntax_Tags::TK_ARROW; }

  /* Relational Intrinsic Operators */
".eq." { return Syntax_Tags::TK_REL_EQ; }
".ne." { return Syntax_Tags::TK_REL_NE; }
".lt." { return Syntax_Tags::TK_REL_LT; }
".le." { return Syntax_Tags::TK_REL_LE; }
".gt." { return Syntax_Tags::TK_REL_GT; }
".ge." { return Syntax_Tags::TK_REL_GE; }
"=="   { return Syntax_Tags::TK_REL_EQ; }
"/="   { return Syntax_Tags::TK_REL_NE; }
"<"    { return Syntax_Tags::TK_REL_LT; }
"<="   { return Syntax_Tags::TK_REL_LE; }
">"    { return Syntax_Tags::TK_REL_GT; }
">="   { return Syntax_Tags::TK_REL_GE; }

  /* Logical Operators */
".not."  { return Syntax_Tags::TK_NOT_OP; }
".and."  { return Syntax_Tags::TK_AND_OP; }
".or."   { return Syntax_Tags::TK_OR_OP; }
".eqv."  { return Syntax_Tags::TK_EQV_OP; }
".neqv." { return Syntax_Tags::TK_NEQV_OP; }

  /* ---------------------------- LITERALS ---------------------------- */

  /* R724: logical-literal-constant (7.4.5) */
".false."(_{KIND})? { return Syntax_Tags::TK_FALSE_CONSTANT; }
".true."(_{KIND})?  { return Syntax_Tags::TK_TRUE_CONSTANT; }

  /* R724: char-literal-constant (7.4.4.3) */
({KIND}_)?\"(\"\"|[^"])*\" /* non-greedy  */ {
   return Syntax_Tags::SG_CHAR_LITERAL_CONSTANT;
} 
    
({KIND}_)?'(''|[^'])*'  /* non-greedy */   { 
   return Syntax_Tags::SG_CHAR_LITERAL_CONSTANT; 
} 
  /* " */

  /* R714: real-literal-constant (7.4.3.2) */
  /* significand [exponent-letter exponent] [_ kind-param] */
{SIGNIFICAND}{EXPONENT}?(_{KIND})? { 
   return Syntax_Tags::SG_REAL_LITERAL_CONSTANT; 
}
  /* digit-string exponent-letter exponent [_ kind-param] */
{DIGIT}+{EXPONENT}(_{KIND})?       { 
   return Syntax_Tags::SG_REAL_LITERAL_CONSTANT; 
}
  /* R708: int-literal-constant (7.4.3.1) */
{DIGIT}+(_{KIND})?            {
   return Syntax_Tags::SG_INT_LITERAL_CONSTANT; 
}


  /* These must be last to allow keyword and operator completion */

  /* Defined operator */
\.[a-z]+\. { return Syntax_Tags::TK_DEF_OP; } 
{NAME}     { return Syntax_Tags::TK_NAME; }

[ \t\r] /* eat whitespace */

. { return Syntax_Tags::UNKNOWN; }

