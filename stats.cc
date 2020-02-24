/***

Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

***/

typedef struct stats_s {
  test_t          operation;
  int             tpi;
  int             size;
  double          time;  /* seconds */
  struct stats_s *next;
} stats_t;

/*
 * Function: stats_find_column
 * Description: Check Whether the task is in pool
 * Para: 
 *   chain: task chain
 *   tpi: thread per instance
 *   size: size of instance
 */ 
bool stats_find_column(stats_t *chain, int tpi, int size) {
  while(chain!=NULL) {
    if(chain->tpi==tpi && chain->size==size)
      return true;
    chain=chain->next;
  }
  return false;
}

/*
 * Function: stats_find
 * Description: Modify the head of chain to the specific task
 * Para: 
 *   chain: task chain
 *   test_t: operation
 *   tpi: thread per instance
 *   size: size of instance
 */ 
stats_t *stats_find(stats_t *chain, test_t test, int tpi, int size) {
  while(chain!=NULL) {
    if(chain->operation==test && (chain->tpi==tpi || chain->tpi==0) && chain->size==size)
      break;
    chain=chain->next;
  }
  return chain;
}

/*
 * Function: stats_report
 * Description: Write the log report into file
 * Para: 
 *   out: file pointer
 *   bool: whether we should output in a pretty mode
 *   chain: task pool
 *   tests: whether the task is performed
 *   first: index of first operation
 *   last: index of last operation
 *   sizes: size of each instance
 *   size_count: total number of instance
 */ 
void stats_report(FILE *out, bool pretty, stats_t *chain, bool *tests, int first, int last, int *sizes, int sizes_count) {
  test_t test;
  if(pretty) {
    fprintf(out, "             ");
    for(int size_index=0;size_index<sizes_count;size_index++) {
      for(int tpi=1;tpi<=32;tpi=2*tpi) {
        if(stats_find_column(chain, tpi, sizes[size_index]))
          if(tpi>1)
            fprintf(out, " %4d:%-2d  ", sizes[size_index], tpi);
          else
            fprintf(out, " %4d     ", sizes[size_index]);
      }
    }
    fprintf(out, "\n");
    for(int test_index=first;test_index<=last;test_index++) {
      if(!tests[test_index-first])
        continue;
      test=static_cast<test_t>(test_index);
      fprintf(out, "%-13s", test_name(test));
      for(int size_index=0;size_index<sizes_count;size_index++) {
        for(int tpi=1;tpi<=32;tpi=tpi*2) {
          if(stats_find(chain, test, tpi, sizes[size_index])==NULL)
            continue;
          //stats_report_throughput(out, pretty, stats_find(chain, test, tpi, sizes[size_index]));
          fprintf(out, "  ");
        }
      }//end inner for
      fprintf(out, "\n");
    }
  }//end pretty
  else {
    fprintf(out, ",");
    for(int index=0;index<sizes_count;index++) {
      fprintf(out, "%d", sizes[index]);
      if(index<sizes_count-1)
        fprintf(out, ",");
    }
    fprintf(out, "\n");
    for(int test_index=first;test_index<=last;test_index++) {
      if(!tests[test_index-first])
        continue;
      test=static_cast<test_t>(test_index);
      fprintf(out, "\"%s\",", test_name(test));
      for(int size_index=0;size_index<sizes_count;size_index++) {
        //stats_report_throughput(out, pretty, stats_find_fastest(chain, test, sizes[size_index]));
        if(size_index<sizes_count-1)
          fprintf(out, ",");
      }
      fprintf(out, "\n");
    }
  }
}