!* 该模块定义了神经网络结构的方法。
module mod_NNStructureSaver
use mod_Precision
use mod_NNStructure
use mod_Log
implicit none

    !-----------------------------
    ! 工作类：保存网络结构的数据 |
    !-----------------------------
    type, public :: NNStructureSaver
        
    !||||||||||||    
    contains   !|
    !||||||||||||

        procedure, public :: preserve => m_save
		procedure, public :: load     => m_load

    end type NNStructureSaver
    !--------------------------------------------------------

    
    !-------------------------
    private :: m_save
	private :: m_load
    !-------------------------

!||||||||||||    
contains   !|
!||||||||||||
    
	!* 保存网络结构数据.
    subroutine m_save( this, net, file_name )
    implicit none
        class(NNStructureSaver), intent(inout) :: this
		class(NNStructure), pointer, intent(in) :: net
		character(len=*), intent(in) :: file_name

		integer :: layer_index, l_count
		
		l_count = net % layers_count
		
		open(UNIT=30, FILE=file_name, &
            ACCESS='stream', FORM='unformatted', STATUS='replace')
			
		do layer_index=1, l_count	
		
			associate (                                      &                              
				W     => net % pt_W(layer_index) % W,        &
                Theta => net % pt_Theta(layer_index) % Theta &
            )
		
			write(30) W 
			write(30) Theta
						
			end associate
			
		end do
			
		close(30)
            
        call LogDebug("NNStructureSaver: SUBROUTINE m_save")

        return
    end subroutine m_save
    !====

	!* 读取网络结构数据
	subroutine m_load( this, net, file_name )
	implicit none
        class(NNStructureSaver), intent(inout) :: this
		class(NNStructure), pointer, intent(in) :: net
		character(len=*), intent(in) :: file_name
		
		integer :: layer_index, l_count
		
		l_count = net % layers_count
		
		open(UNIT=30, FILE=file_name, &
            ACCESS='stream', FORM='unformatted', STATUS='old')
			
		do layer_index=1, l_count	
		
			associate (                                      &                              
				W     => net % pt_W(layer_index) % W,        &
                Theta => net % pt_Theta(layer_index) % Theta &
            )
		
			read(30) W 
			read(30) Theta
						
			end associate
			
		end do
			
		close(30)
            
        call LogDebug("NNStructureSaver: SUBROUTINE m_load")
		
		return
	end subroutine m_load
	!====
    
end module