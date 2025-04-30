import traceback
import sys

class CustomException(Exception):   #We need teh predefined exception also from python so we inherit from it.
    def __init__(self,error_message,error_detail:sys):  #error_detail is the sys module.
        super().__init__(error_message)     #By inheriting from Exception, your CustomException will automatically behave like a regular exception. So you can raise it, try and except it just like any other exception.
        self.error_message = self.get_detailed_error_message(error_message,error_detail)
        
    @staticmethod
    def get_detailed_error_message(error_message,error_detail:sys):
        _,_,exc_tb = traceback.sys.exc_info()    #We only need the last thing which is the traceback so _,_,exec_tb
        filename=exc_tb.tb_frame.f_code.co_filename  #Filename where the error occured.
        line_number = exc_tb.tb_lineno #Line number where the error occured.
        return f"Error occured in script: [{filename}] at line number: [{line_number}] error message: [{error_message}]"
        
    def __str__(self):  #Text representation of error message To get this "Error occured in script: [{filename}] at line number: [{line_number}] error message: [{error_message}]" or else it would just print the error message. Not the filename and line number
        return self.error_message
            
            
            
        
    