/*
   Copyright (c) 2019, Triad National Security, LLC. All rights reserved.

   This is open source software; you can redistribute it and/or modify it
   under the terms of the BSD-3 License. If software is modified to produce
   derivative works, such modified software should be clearly marked, so as
   not to confuse it with the version available from LANL. Full text of the
   BSD-3 License can be found in the LICENSE file of the repository.
*/

/**
  \file subtender.hh

  \brief Declaration of function interfaces exposed by subtender.cc

  Demonstration program to inventory subroutine names and calls and match
  them against common external libraries from which source code may have been
  imported. This assists with managing legacy code by detecting code that may
  be externally maintained and with intellectual property protection
*/

#ifndef MARK_EXECUTABLE_HH
#define MARK_EXECUTABLE_HH 1

#include "flpr/flpr.hh"
#include <iostream>
#include <string>
#include <set>

/*--------------------------------------------------------------------------*/

using File = FLPR::Parsed_File<>;
using Cursor = typename File::Parse_Tree::cursor_t;
using Procedure = FLPR::Procedure<File>;

using set_str = std::set<std::string>;
// using vec_str = std::vector<std::string>;

/**
 *  \brief Subroutine scanner configuration data
 */
struct SubScanConfigurator {
  /// Dry-run mode
  bool dry_run {false};

  /// Source files to scan
  set_str srcfiles;

  /// Subroutines to avoid scanning
  set_str excluded_subs;

  /** Libraries to scan; if specified, only explicitly listed libraries will be
   * scanned
   */
  set_str included_libs;

  /** \brief Libraries to avoid scanning; libraries to scan are defaults minus
   * excludes
   */
  set_str excluded_libs;

  /// Libraries to scan by default
  set_str default_libs;

  /// Libraries to scan
  set_str scan_libs;

  /// Constructor, bare
  SubScanConfigurator()
  {
  }

  /// Constructor
  SubScanConfigurator(set_str SourceFiles,
                      set_str ExcludedSubs,
                      set_str IncludedLibs,
                      set_str ExcludedLibs) :
                      srcfiles(SourceFiles),
                      excluded_subs(ExcludedSubs),
                      included_libs(IncludedLibs),
                      excluded_libs(ExcludedLibs)
  {
  }
}; // SubScanConfigurator
/*--------------------------------------------------------------------------*/

void print_usage(std::ostream &os);
void print_usage(std::ostream &os,
                 const SubScanConfigurator& sscfg);
void parse_cmd_line(int argc,
                    char* const* argv,
                    SubScanConfigurator& sscfg);
bool detect_subroutine(const std::string& lname,
                       const std::string& sublib);
bool subtender_procedure(File &file,
                         Cursor c,
                         bool internal_procedure,
                         bool module_procedure);
bool exclude_procedure(Procedure const &subp);
bool subtender_file(std::string const &filename);
void write_file(std::ostream &os,
                File const &f);

#endif
