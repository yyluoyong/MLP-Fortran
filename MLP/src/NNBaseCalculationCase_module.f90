module mod_BaseCalculationCase
implicit none
    
!------------------
! �����ࣺ�������� |
!------------------
type, abstract, public :: BaseCalculationCase

!||||||||||||    
contains   !|
!||||||||||||

    !* ����������������
    procedure(abs_main), deferred, public :: main

end type BaseCalculationCase
!===================
    

!------------------
! �����ࣺ�����ӿ� |
!------------------	
abstract interface   

	!* ������
	subroutine abs_main( this )
    import :: BaseCalculationCase
	implicit none
		class(BaseCalculationCase), intent(inout) :: this

	end subroutine
	!====

end interface
!===================
    
end module