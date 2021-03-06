/*
   Copyright (c) 2019, Triad National Security, LLC. All rights reserved.

   This is open source software; you can redistribute it and/or modify it
   under the terms of the BSD-3 License. If software is modified to produce
   derivative works, such modified software should be clearly marked, so as
   not to confuse it with the version available from LANL. Full text of the
   BSD-3 License can be found in the LICENSE file of the repository.
*/

#include "LL_Helper.hh"
#include "flpr/Prgm_Parsers.hh"
#include "flpr/Prgm_Tree.hh"
#include "test_helpers.hh"
#include <iostream>

using namespace FLPR;

using PS = FLPR::Prgm::Parsers<FLPR::Prgm::Prgm_Node_Data>;
// clang-format off

bool test_instantiate() {
  LL_Helper ls({"subroutine foo",
                  "use foo",
                  "use, intrinsic :: iso_c_binding",
                  "import, none",
                "100 format(foo)",
                  "parameter(a = 3)",
                  "implicit none",
                  "implicit integer(a)",
                  "implicit real(b-z)",
                  "integer i",
                  "do i = 1, a",
                    "if(i == a) then",
                      "return",
                    "end if",
                  "enddo",
                  "where (a>3)",
                    "a=3",
                  "endwhere",
                "end",
                "function bar",
                  "integer i",
                  "do i = 1,100",
                  "enddo",
                "return",
                "contains", "subroutine h",
                            "end subroutine h",
                "end"});
  PS::State state(ls.ll_stmts());
  auto res = PS::program(state);
  return res.match;
}

bool labeled_do() {
  LL_Helper ls({"function foo",
                "integer i", 
                "do 500 i=1,5",
                "500 continue",
                "end function"});
  PS::State state(ls.ll_stmts());
  auto res = PS::program(state);
  return res.match;
}

// Multiple labeled-do's using the same exit point
bool labeled_do2() {
  LL_Helper ls({"function foo",
                "integer i,j", 
                "do 500 i=1,5",
                "do 500 j=1,5",
                "500 continue",
                "end function"});
  PS::State state(ls.ll_stmts());
  auto res = PS::program(state);
  return res.match;
}

// Single labeled do with labeled end-do-stmt
bool labeled_do3() {
  LL_Helper ls({"function foo",
                "integer i", 
                "do 500 i=1,5",
                "500 enddo",
                "end function"});
  PS::State state(ls.ll_stmts());
  auto res = PS::program(state);
  return res.match;
}

bool derived_type_def() {
  LL_Helper ls({"type t1",
                " type(t2) :: v(0:n)",
                "end type t1"});
  PS::State state(ls.ll_stmts());
  auto res = PS::derived_type_def(state);
  return res.match;
}
  
bool block_construct() {
  LL_Helper ls({"block",
                " integer a",
                " a=3",
                "end block"});
  PS::State state(ls.ll_stmts());
  auto res = PS::block_construct(state);
  return res.match;
}

bool do_select_construct() {
  LL_Helper ls({"SCAN_LINE: DO I = 1, 80",
                "  CHECK_PARENS: SELECT CASE (LINE (I:I))",
                "  CASE ('(')",
                "    LEVEL = LEVEL + 1",
                "  CASE (')')",
                "    LEVEL = LEVEL - 1",
                "    IF (LEVEL < 0) THEN",
                "      PRINT *, 'UNEXPECTED RIGHT PARENTHESIS'",
                "      EXIT SCAN_LINE",
                "    END IF",
                "  CASE DEFAULT",
                "    ! Ignore all other characters",
                "  END SELECT CHECK_PARENS",
                "END DO SCAN_LINE"});
  PS::State state(ls.ll_stmts());
  auto res = PS::do_construct(state);
  return res.match;
}


bool module_program() {
  LL_Helper ls({"module a",
                "end module a",
                "program b",
                "end program b"});
  PS::State state(ls.ll_stmts());
  auto res = PS::program(state);
  return res.match;
}

// clang-format on

int main() {
  TEST_MAIN_DECL;
  TEST(test_instantiate);
  TEST(block_construct);
  TEST(labeled_do);
  TEST(labeled_do2);
  TEST(labeled_do3);
  TEST(derived_type_def);
  TEST(do_select_construct);
  TEST(module_program);
  TEST_MAIN_REPORT;
}
