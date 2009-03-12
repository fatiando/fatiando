/* 
 * File:   log.cpp
 * $Revision$
 * Last edited: $Date$
 * Edited by: $Author$
 *
 * Created by: Leonardo Uieda (leouieda@gmail.com)
 * 
 * Created on March 11, 2009, 6:40 PM
 *
 * Description:
 *  Method implementations for the Log class.
 */

#include <time.h>
#include "headers/log.h"

/* FIELDS */
bool Log::started = false;
std::vector<std::string>* Log::log = new std::vector<std::string>;


/* CONSTRUCTORS AND DESTRUCTOR */
Log::Log()
{
}


Log::~Log()
{
}


/* METHODS */
void Log::start(const char *program_name)
{
/* **************************************************************************
 * Starts the log for all instances of the Log class. Also marks the time when
 * this happened. If the log has already been started, this command will be
 * ingnored.
 **************************************************************************** */

    if(Log::started == false)
    {
        /* Mark the date and time when the log started */
        time_t rawtime;
        struct tm * timeinfo;
        time ( &rawtime );
        timeinfo = localtime ( &rawtime );

        /* Start the log with a nice header */
        std::string msg;

        msg.assign("Hello! This is a log file for program ");
        msg.append(program_name);
        msg.append(".");
        Log::log->push_back(msg);

        msg.assign(asctime (timeinfo));
        Log::log->push_back(msg);

        msg.assign("");
        Log::log->push_back(msg);

        /* Mark that the log has started */
        Log::started = true;
    }
}


bool Log::log_started()
{
/* **************************************************************************
 * Returns true if the log was already started and false if it wasn't.
 **************************************************************************** */
    return Log::started;
}


void Log::append(const char *new_msg) throw (LogNotStartedException)
{
/* **************************************************************************
 * Skips a line and appends new_msg to the log.
 * Throws a LogNotStartedException if trying to do this before starting the log.
 **************************************************************************** */

    if(Log::started == false)
    {
        LogNotStartedException e("append message to");
        throw e;
    }
    
    std::string msg(new_msg);
    Log::log->push_back(msg);
}


void Log::print(FILE* log_file) throw (LogNotStartedException, InvalidLogFileException)
{
/* **************************************************************************
 * Creates the file file_name and prints the log message to it.
 * Throws a LogNotStartedException if trying to do this before starting the log.
 * Throws a InvalidLogFileException when the FILE* passed is NULL.
 **************************************************************************** */

    if(Log::started == false)
    {
        LogNotStartedException e("print");
        throw e;
    }

    /* Check if the FILE* actually points to a file */
    if(log_file == NULL)
    {
        InvalidLogFileException e;
        throw e;
    }
    
    /* Print the log to the file */
    unsigned int l; /* Line counter */
    for(l=0; l < Log::log->size(); l++)
    {
        fprintf(log_file, "%s\n", Log::log->at(l).c_str());
    }
}
