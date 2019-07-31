/*
   Copyright (c) 2019, Triad National Security, LLC. All rights reserved.

   This is open source software; you can redistribute it and/or modify it
   under the terms of the BSD-3 License. If software is modified to produce
   derivative works, such modified software should be clearly marked, so as
   not to confuse it with the version available from LANL. Full text of the
   BSD-3 License can be found in the LICENSE file of the repository.
*/

#include "flpr/parse_stmt.hh"
#include "parse_helpers.hh"

using namespace FLPR;
using FLPR::Stmt::Stmt_Tree;

bool test_allocate_coarray_spec() {
  // No brackets
  TSS(allocate_coarray_spec, "*");
  // The first 1 is to confuse the label parser
  TSS(allocate_coarray_spec, "1 1,*");
  TSS(allocate_coarray_spec, "1 1,2,3,*");
  TSS(allocate_coarray_spec, "1 1:1,2,3,*");
  TSS(allocate_coarray_spec, "1 1:1,1:2,1:3,1:*");
  return true;
}

bool actual_arg_spec() {
  TPS(actual_arg_spec, "a+7, p=foo", TK_COMMA);
  return true;
}

bool array_element() {
  TPS(array_element, "a(1)", BAD);
  TPS(array_element, "a(1,2)", BAD);
  TPS(array_element, "a(b)(1)", TK_PARENL);
  TPS(array_element, "c%a(b)(1)", TK_PARENL);
  TPS(array_element, "c(2)%d%a(b)(1)", TK_PARENL);
  TPS(array_element, "c(3)%a(b)(1)", TK_PARENL);
  FPS(array_element, "a(1:3)", TK_NAME);
  FPS(array_element, "a=", TK_NAME);
  FPS(array_element, "a%b", TK_NAME);
  TPS(array_element, "c(2)[5,2,STAT=var]%d[1]%a(b)[3](1)", TK_PARENL);
  return true;
}

bool test_expr() {
  TPS(expr, "a", BAD);
  TPS(expr, "'a b c',", TK_COMMA);
  TPS(expr, "\"a b c\",", TK_COMMA);
  TPS(expr, "','", BAD);
  TPS(expr, "a==3,", TK_COMMA);
  FPS(expr, "a=3", TK_NAME);
  TPS(expr, "a+b(3,[:])", BAD);
  TPS(expr, "a+b(3,[:]))", TK_PARENR);
  TPS(expr, "a+b(3,[:])]", TK_BRACKETR);
  TPS(expr, "a+b%q(3,[:]),]", TK_COMMA);
  TPS(expr, "a**3+b(3,[:]):]", TK_COLON);
  FPS(expr, ":", TK_COLON);
  return true;
}

bool test_image_selector() {
  TPS(image_selector, "[1]", BAD);
  TPS(image_selector, "[a+1]", BAD);
  TPS(image_selector, "[a+1,b-1]", BAD);
  TPS(image_selector, "[a+1,b-1,stat=a]", BAD);
  TPS(image_selector, "[2,3,4,stat=a]", BAD);
  TPS(image_selector, "[2+2,team=3,stat=a]", BAD);
  return true;
}

bool proc_component_ref() {
  TPS(proc_component_ref, "a % b =>", TK_ARROW);
  return true;
}

bool test_variable() {
  TPS(variable, "a", BAD);
  TPS(variable, "a+", TK_PLUS);
  TPS(function_reference, "a()=", TK_EQUAL);
  TPS(variable, "a()=", TK_EQUAL);
  TPS(variable, "a%b(:)=", TK_EQUAL);
  TPS(variable, "b[2,3,4,stat=var]=", TK_EQUAL);
  return true;
}

int main() {
  TEST_MAIN_DECL;
  TEST(test_allocate_coarray_spec);
  TEST(actual_arg_spec);
  TEST(array_element);
  TEST(test_expr);
  TEST(test_image_selector);
  TEST(proc_component_ref);
  TEST(test_variable);
  TEST_MAIN_REPORT;
}