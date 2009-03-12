/* 
 * File:   log.h
 * $Revision$
 * Last edited: $Date $
 * Edited by: $Author $
 *
 * Created by: Leonardo Uieda (leouieda@gmail.com)
 *
 * Created on March 11, 2009, 6:40 PM
 *
 * Description:
 *  This file contains the Log class and related exceptions.
 *  This is a class that keeps the messages a program want's to print to a .log
 *  file. The log message created is static and therefore there is only one
 *  for the whole program. This is so that different instances of the Log class
 *  will edit the sabe log message. This eliminates the need to make a global
 *  Log instance.
 *  Usage:
 *
 *      Before the class can be used:
 *          - Create and instance of Log;
 *          - Call start(const char* program_name);
 *
 *      Now the log is ready to be used. Anywhere in the program simply:
 *          - Create and instance of Log;
 *          - Call append(const char* new_msg);
 *
 *      You can check wether the log was started (with the start method) by
 *      calling the log_started() method.
 *
 *      When you want to print the log to a file call the print(FILE* log_file)
 *      method from any instance of Log.
 */

#ifndef _LOG_H
#define	_LOG_H

#include <stdio.h>
#include <string>
#include <vector>
#include <exception>

class LogNotStartedException: public std::exception
{
/* **************************************************************************
 * This is thrown when the append or print methods are called before the log
 * was started.
 **************************************************************************** */

    protected:
        std::string msg;

    public:
        LogNotStartedException()
        {
            msg = "\nERROR! Tried access Log class instance's methods before starting log.\n";
        }
        LogNotStartedException(const char* function) 
        {
            std::string func(function);
            msg = "\nERROR! Tried to ";
            msg += func;
            msg += " log before it was started.\n";
        }
        ~LogNotStartedException() throw ()
        {
        }
        virtual const char* what() const throw()
        {
            return msg.c_str();
        }
};


class InvalidLogFileException: public std::exception
{
/* **************************************************************************
 * This is thrown when there is an error with the log FILE* passed to the print
 * method.
 **************************************************************************** */

    protected:
        std::string msg;

    public:
        InvalidLogFileException() {
            msg = "\nERROR! Can't print log! Invalid file pointer passed.\n";
        }
        ~InvalidLogFileException() throw ()
        {
        }
        virtual const char* what() const throw()
        {
            return msg.c_str();
        }
};


class Log {
/* **************************************************************************
 * The Log class.
 * This is a class that keeps the messages a program want's to print to a .log
 * file. The log message created is static and therefore there is only one
 * for the whole program.
 * Before starting to append log messages, the log must be started using the
 * Log::start(const char *program_name) method.
 * The Log::append(const char *new_msg) method appends a new line to the log
 * file containing the string new_msg.
 * To print the log to a file, use Log::print(const char *file_name) and the log
 * will be printed to file_name (pass with extension. Ex: bla.log).
 **************************************************************************** */

    private:
        static std::vector<std::string> *log;
        static bool started;        

    public:
        Log();
        ~Log();
        static void start(const char *program_name);
        static bool log_started();
        static void append(const char *new_msg) throw (LogNotStartedException);
        static void print(FILE* log_file) throw (LogNotStartedException, InvalidLogFileException);
};

#endif	/* _LOG_H */

