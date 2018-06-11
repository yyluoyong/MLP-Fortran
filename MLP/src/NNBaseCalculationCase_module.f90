module mod_BaseCalculationCase
implicit none
    
!------------------
! 抽象类：计算算例 |
!------------------
type, abstract, public :: BaseCalculationCase

!||||||||||||    
contains   !|
!||||||||||||

    !* 计算算例的主函数
    procedure(abs_main), deferred, public :: main

end type BaseCalculationCase
!===================
    

!------------------
! 抽象类：函数接口 |
!------------------	
abstract interface   

	!* 主函数
	subroutine abs_main( this )
    import :: BaseCalculationCase
	implicit none
		class(BaseCalculationCase), intent(inout) :: this

	end subroutine
	!====

end interface
!===================
    
end module