!===============================================================================
! Program: hfradar_main
! Description: Executive routine for High-Frequency (HF) Radar data processing.
!              This program handles the iterative ingestion of radial velocity 
!              fields and manages the synthesis of total vector current fields 
!              onto a computational grid.
!===============================================================================
program hfradar_main
    use HFradar_data
    implicit none
    
    ! --- Object Declarations and Memory Allocation ---
    type(HFRadarProcessor) :: processor   ! High-level processing class instance
    character(len=500) :: input_file      ! Path for filtered radial input files
    character(len=500) :: output_file     ! Path for synthesized total vector outputs
    character(len=500) :: grid_file       ! NetCDF file containing the geographic grid
    real(8) :: max_distance               ! Search radius for spatial interpolation (km)
    integer :: i                          ! Temporal/Index loop counter
    character(len=3) :: i_str             ! String buffer for file indexing
    
    ! --- Computational Performance Monitoring ---
    real(8) :: start_time, end_time, iteration_time
    real(8) :: total_start, total_end, total_time
    
    ! Initialize global CPU timer
    call cpu_time(total_start)
    
    ! --- Processing Loop ---
    ! Iterate through specific data subsets or time steps
    do i = 0, 263
   
        call cpu_time(start_time)
        
        ! Convert loop index to 3-digit formatted string for file I/O
        write(i_str, '(I3.3)') i
        
        ! Construct input path for filtered radial data (January 2026 dataset)
        input_file = "../../data/january_2026/radials_10_days/filtered_radials/" // &
                     "filtered_radials6_" // trim(i_str) // ".txt"
        
        ! Construct output path for processed total current vectors
        output_file = "../../data/january_2026/totals_10_days/" // &
                      "medsea_totals_" // trim(i_str) // "_all_grid.txt"
                      
        ! Define the static NetCDF grid metadata file
        grid_file = "../../data/" // &
                    "hfradar_totals_grid_icatmar.nc"
        
        ! Set the spatial constraint for the radial-to-total synthesis algorithm
        max_distance = 6.0d0
        
        ! Execute computational kernels:
        ! 1. Parse text-based radial files
        ! 2. Perform spatial mapping to the NetCDF grid
        ! 3. Export results to the specified output file
        call processor%process_text_files(input_file, output_file, grid_file, max_distance)
        
        ! Memory Management: Release allocated arrays and reset processor state
        call processor%cleanup()
        
        ! Calculate and log iteration performance metrics
        call cpu_time(end_time)
        iteration_time = end_time - start_time
        
        print '(A,I3,A,F10.3,A)', "Iteration ", i, " completed in ", iteration_time, " seconds"
        
    end do
    
    ! --- Final Execution Analytics ---
    call cpu_time(total_end)
    total_time = total_end - total_start
    
    print *
    print *, "=========================================="
    print '(A,F10.3,A)', "Total Program Execution Time:   ", total_time, " seconds"
    print '(A,F10.3,A)', "Average Latency per Iteration:  ", total_time/3.0d0, " seconds"
    print *, "=========================================="
    print *
    print *, "Program execution terminated successfully."
    
end program hfradar_main
