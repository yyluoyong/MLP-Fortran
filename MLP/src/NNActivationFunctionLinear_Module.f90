module mod_Linear
use mod_Precision
use mod_BaseActivationFunction
implicit none    

!-------------------
! �����ࣺ����� |
!-------------------
type, extends(BaseActivationFunction), public :: Linear
    !* �̳���BaseActivationFunction��ʵ����ӿ�

!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: f       => m_fun_Linear
    procedure, public :: f_vect  => m_fun_Linear_vect 
    procedure, public :: df      => m_df_Linear
    procedure, public :: df_vect => m_df_Linear_vect

end type Linear
!===================

    !-------------------------
    private :: m_fun_Linear
    private :: m_df_Linear
    private :: m_fun_Linear_vect
	private :: m_df_Linear_vect
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* Linear����
    subroutine m_fun_Linear( this, index, x, y )
    implicit none
        class(Linear), intent(inout) :: this
        integer, intent(in) :: index
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), intent(out) :: y
    
        y = x(index)
    
        return
    end subroutine
    !====
    
    !* �������������Linear����
	subroutine m_fun_Linear_vect( this, x, y )
	implicit none
        class(Linear), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: y	
	
        y = x
        
		return
	end subroutine
	!====
    
	!* Linear������һ�׵���
	subroutine m_df_Linear( this, index, x, dy )
	implicit none
        class(Linear), intent(inout) :: this
		integer, intent(in) :: index
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), intent(out) :: dy
	
		dy = 1
	
		return
	end subroutine
	!====
	
	!* �������������Linear������һ�׵���
	subroutine m_df_Linear_vect( this, x, dy )
	implicit none
        class(Linear), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: dy
	
		dy = 1
	
		return
	end subroutine
	!====

end module