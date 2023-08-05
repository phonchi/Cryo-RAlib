#*******************************************************************************
#Author (C) 2019, Fabian Schoenfeld (fabian.schoenfeld@mpi-dortmund.mpg.de)
#Copyright (C) 2019, Max Planck Institute of Molecular Physiology, Dortmund

#   This program is free software: you can redistribute it and/or modify it 
#under the terms of the GNU General Public License as published by the Free 
#Software Foundation, either version 3 of the License, or (at your option) any
#later version.

#   This program is distributed in the hope that it will be useful, but WITHOUT
#ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License along with
#this program.  If not, please visit: http://www.gnu.org/licenses/
#*******************************************************************************

from __future__ import print_function

import os
import sys
import subprocess

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

def main(args):

    # check SPHIRE
    try:
        import EMAN2
        import sparx
        import global_def
        ver = global_def.SPARXVERSION
    
    except Exception as err:
        print( "\nSPHIRE available: .................................................... FAIL" )
        print( "    " + str(err))
        return EXIT_FAILURE
    
    print( "\nSPHIRE installation: .................................................check" )
    print( "    " + ver )
    
    # check MPI
    try:
        ver = subprocess.check_output( "conda list pydusa", shell=True, stderr=subprocess.STDOUT )
        ver = str(ver.decode('utf-8'))
        ver = [ entry for entry in ver.split("\n") if not "#" in entry and entry != "" ][0]
        ver = " ".join(ver.split())
    
        import mpi
        mpi.mpi_init(0, [])
        mpi.mpi_finalize()
    
    except Exception as err:
        print( "\nMPI available: ....................................................... FAIL" )
        print( "    " + str(err))
        print( "    conda list pydusa --> ", ver )
    
        return EXIT_FAILURE
    
    print( "\nMPI available: .......................................................check" )
    print( "    " + ver )

    # check CUDA installation
    try:
        ver = subprocess.check_output( "nvcc --version", shell=True, stderr=subprocess.STDOUT )
        ver = str(ver.decode('utf-8'))
        ver = [ entry for entry in ver.split("\n") if entry != "" ]
    
    except subprocess.CalledProcessError as err:
        print( "\nCUDA available: ...................................................... FAIL" )
        print( "    \""+err.cmd+"\" failed with:\n    ", err.output)
        
        # check path variables
        path = subprocess.check_output( "echo $PATH", shell=True )
        path = path.split(":")
        print( "    $PATH:" )
        for i in path: print("    "+i)
        
        path = subprocess.check_output( "echo $LD_LIBRARY_PATH", shell=True )
        path = path.split(":")
        print( "    $LD_LIBRARY_PATH:" )
        for i in path: print("    "+i)
        
        # check system installation
        ldconfig = subprocess.check_output( "ldconfig -p | grep cuda", shell=True ).split("\n")
        print( "    ldconfig -p | grep cuda:" )
        for i in ldconfig: print("    "+i.lstrip())
        
        return EXIT_FAILURE
        
    print( "\nCUDA available: ......................................................check" )
    for i in ver: print( "    " + i )

    # all done
    print( "\nEverything looks good, good luck with your project! :)\n" )
    
    return EXIT_SUCCESS

if __name__=="__main__":
    sys.exit( main(sys.argv) )
