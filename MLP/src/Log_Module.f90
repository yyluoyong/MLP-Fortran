module mod_Log
implicit none
    
	!-----------------------------
	!* 日志文件的前缀，默认为空
	character(len=50), save, private :: file_name_prefix = ''	
	character(len=4),  save, private :: file_format = '.txt'
	character(len=20), save, private :: log_path = './Message/'
	
	character(len=5), save, private :: error_file_name_suffix = 'ERROR'
	character(len=4), save, private :: info_file_name_suffix = 'INFO'
	character(len=5), save, private :: debug_file_name_suffix = 'DEBUG'
	
    integer, parameter :: FILE_NAME_LEN_MAX = 180
    character(len=FILE_NAME_LEN_MAX), save, private :: error_file_name
    character(len=FILE_NAME_LEN_MAX), save, private :: info_file_name
    character(len=FILE_NAME_LEN_MAX), save, private :: debug_file_name
	
	!* 文件名是否已经生成
	logical, save, private :: is_file_name_generation = .false.
    
    logical, save, private :: is_debug = .false.
    !-----------------------------
	
	
	!-----------------------------
	private :: m_generate_file_name
    private :: m_output_string
	private :: m_get_date_str
	!-----------------------------

!||||||||||||    
contains   !|
!||||||||||||
	
	!* 设置日志文件的前缀
	subroutine Log_set_file_name_prefix( prefix )
	implicit none
		character(len=*), intent(in) :: prefix
		
		file_name_prefix = TRIM(ADJUSTL(prefix))
		is_file_name_generation = .false.
	
		return
	end subroutine
	!====

    !* 输出调试信息
    subroutine LogDebug( debugInfo )
    implicit none
        character(len=*), intent(in) :: debugInfo

        if (is_debug) then
			call m_generate_file_name()
            write(*, *) debugInfo
            call m_output_string( debug_file_name, debugInfo )
        end if

        return
    end subroutine LogDebug
    !====
    
    
    !* 输出错误信息
    subroutine LogErr( err )
    implicit none
        character(len=*), intent(in) :: err

        write(*,*) "LogErr: Error! See ERROR.txt for details."

		call m_generate_file_name()
        call m_output_string( error_file_name, err )

        return
    end subroutine LogErr
    !====

    !* 输出信息
    subroutine LogInfo( info )
    implicit none
        character(len=*), intent(in) :: info

		call m_generate_file_name()
        write(*, *) TRIM(ADJUSTL(info))
        call m_output_string( info_file_name, info )

        return
    end subroutine LogInfo
    !====

	
	!* 将字符串输出到指定文件
    subroutine m_generate_file_name()
    implicit none
        
		character(len=20) :: date_str
		
		if( is_file_name_generation )  return
		
		call m_get_date_str( date_str )

		if (TRIM(ADJUSTL(file_name_prefix)) == '') then
			error_file_name = TRIM(ADJUSTL(log_path))
			info_file_name  = TRIM(ADJUSTL(log_path))
			debug_file_name = TRIM(ADJUSTL(log_path))
		else
			error_file_name = TRIM(ADJUSTL(log_path)) // &
				TRIM(ADJUSTL(file_name_prefix)) // "-"
			info_file_name  = TRIM(ADJUSTL(log_path)) // &
				TRIM(ADJUSTL(file_name_prefix)) // "-"
			debug_file_name = TRIM(ADJUSTL(log_path)) // &
				TRIM(ADJUSTL(file_name_prefix)) // "-"
		end if
		
		error_file_name = TRIM(ADJUSTL(error_file_name))  // &
			TRIM(ADJUSTL(date_str)) // "-"                // &              
			TRIM(ADJUSTL(error_file_name_suffix))         // &
			TRIM(ADJUSTL(file_format))
			
		info_file_name  = TRIM(ADJUSTL(info_file_name))   // &
			TRIM(ADJUSTL(date_str)) // "-"                // &                 
			TRIM(ADJUSTL(info_file_name_suffix))          // &
			TRIM(ADJUSTL(file_format))
			
		debug_file_name = TRIM(ADJUSTL(debug_file_name))  // &
			TRIM(ADJUSTL(date_str)) // "-"                // &                  
			TRIM(ADJUSTL(debug_file_name_suffix))         // &
			TRIM(ADJUSTL(file_format))
		
		is_file_name_generation = .true.
		
        return
    end subroutine m_generate_file_name
    !====

    !* 将字符串输出到指定文件
    subroutine m_output_string( file_name, str )
    implicit none
        character(len=*), intent(in) :: file_name, str

        logical :: alive

        INQUIRE(file=TRIM(ADJUSTL(file_name)), exist=alive)
        if( .not. alive ) then
            open(unit=33, file=TRIM(ADJUSTL(file_name)), &
                form='formatted', status='new')
        else
            open(unit=33, file=TRIM(ADJUSTL(file_name)), &
                form='formatted', status='old', access='append')
        end if

        write(33, *) TRIM(ADJUSTL(str))
        
        close(33)

        return
    end subroutine m_output_string
    !====

	!* 将系统日期时间转换成字符串，
	!* 如: 2018.06.27-12:00，到分钟
	subroutine m_get_date_str( date_str )
    implicit none
        character(len=*), intent(out) :: date_str

        character(len=8) :: YYYYMMDD
		character(len=10) :: HHmmSS_SSS

		call DATE_AND_TIME(YYYYMMDD, HHmmSS_SSS)
		
		date_str = YYYYMMDD(1:4) // "." // &
				   YYYYMMDD(5:6) // "." // &
				   YYYYMMDD(7:8) // "-" // &
				   HHmmSS_SSS(1:2) // "." // &
				   HHmmSS_SSS(3:4)
		
        return
    end subroutine m_get_date_str
    !====

end module mod_Log
