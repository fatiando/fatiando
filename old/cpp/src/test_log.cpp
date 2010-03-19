/* 
 * File:   test_log.cpp
 * $Revision$
 * Last edited: $Date$
 * Edited by: $Author$
 *
 * Created by: Leonardo Uieda (leouieda@gmail.com)
 *
 * Created on March 11, 2009, 7:57 PM
 *
 * Description:
 *  Program for testing the Log class.
 */

#include <stdio.h>
#include <exception>
#include "headers/log.h"
using namespace std;

void log_from_func(){
    Log lff;
    lff.append("Log from a function: PASSED!!!");
}


int main(int argc, char** argv) {

    int pass = 0, fail = 0;

    printf("\nTESTING LOG CLASS...\n");

    Log log;
    FILE *invalid_file = NULL;

    /* TEST 1 */
    printf("\n  1) Log Not Started Exception: ");
    printf("\n    i) append: ");
    try {
        Log tmp_log1;
        tmp_log1.append("LogNotStartedException (append): FAILED");
        printf("Failed!\n");
        fail++;
    } catch(exception& e) {
        printf("Passed!\n    %s", e.what());
        pass++;
    }
    printf("\n    ii) print: ");
    try {
        Log tmp_log2;
        tmp_log2.print(stdout);
        printf("Failed!\n");
        fail++;
    } catch(exception& e) {
        printf("Passed!\n    %s", e.what());
        pass++;
    }

    log.start("Test_Log");

    /* TEST 2 */
    printf("\n  2) Log started by other instance: ");
    printf("\n    i) append: ");
    try {
        Log tmp_log3;
        tmp_log3.append("Log started by other instance (append): PASSED");
        printf("Passed!\n");
        pass++;
    } catch(exception& e) {
        printf("Failed!\n    %s", e.what());
        fail++;
    }
    printf("\n    ii) print: \n");
    try {
        Log tmp_log4;
        tmp_log4.print(stdout);
        printf("\nLog started by other instance (print): Passed!\n");
        pass++;
    } catch(exception& e) {
        printf("Failed!\n    %s", e.what());
        fail++;
    }


    /* TEST 3 */
    printf("\n  3) Invalid Log File Exception: ");
    try {
        Log tmp_log5;
        tmp_log5.print(invalid_file);
        printf("Failed!\n");
        fail++;
    } catch(exception& e) {
        printf("Passed!\n    %s", e.what());
        pass++;
    }

    /* TEST 4 */
    printf("\n  4) Log from a function: \n");    
    try {
        log_from_func();
        Log tmp_log6;
        tmp_log6.print(stdout);
    } catch(exception& e) {
        printf("%s", e.what());

    }

    printf("\n\nSUMARY:\n   Total=%d  Passed=%d  Failed=%d\n\n", pass+fail, pass, fail);
    
    return 0;
}

