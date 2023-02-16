BINARY = main
CXX = g++
CXXFLAGS = -std=c++17
                                                                               
.PHONY : all                                                                    
	all: $(BINARY)                                                              
	                                                                               
$(BINARY) : main.cpp hessian.h field.h                                                            
	    $(CXX) -o $@ $^ $(CXXFLAGS)                                                
		                                                                               
run:                                                                           
	    ./hessian
		                                                                               
.PHONY: clean
clean:                                                                         
	    rm $(BINARY) ./*.o
		                                                                               
.PHONY: all clean                                                              
