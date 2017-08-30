/* File: cache.h */
/*
  This file is a part of the corrfunc package
  Copyright (C) 2015-- Manodeep Sinha (manodeep@gmail.com)
  License: MIT LICENSE. See LICENSE file under the top-level
  directory at https://github.com/manodeep/corrfunc/
*/

#pragma once


#ifdef __cplusplus
extern "C" {
#endif

/*Taken from http://stackoverflow.com/questions/794632/programmatically-get-the-cache-line-size*/
#include <stddef.h>
size_t cache_line_size(void);
    
#ifdef __cplusplus
}
#endif
