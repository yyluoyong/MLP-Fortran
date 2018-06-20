module mod_Sigmod
use mod_Precision
use mod_BaseActivationFunction
implicit none    

!-------------------
! �����ࣺ����� |
!-------------------
type, extends(BaseActivationFunction), public :: Sigmod
    !* �̳���BaseActivationFunction��ʵ����ӿ�

!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: f       => m_fun_sigmod
    procedure, public :: f_vect  => m_fun_sigmod_vect 
    procedure, public :: df      => m_df_sigmod
    procedure, public :: df_vect => m_df_sigmod_vect

end type Sigmod
!===================

    !-------------------------
    private :: m_fun_sigmod
    private :: m_df_sigmod
    private :: m_fun_sigmod_vect
	private :: m_df_sigmod_vect
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* sigmod����
    subroutine m_fun_sigmod( this, index, x, y )
    implicit none
        class(Sigmod), intent(inout) :: this
        integer, intent(in) :: index
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), intent(out) :: y
    
        y = 1.0 / (1.0 + exp(-x(index)))
        
        return
    end subroutine
    !====
    
    !* �������������sigmod����
	subroutine m_fun_sigmod_vect( this, x, y )
	implicit none
        class(Sigmod), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: y
	
		y = 1.0 / (1.0 + exp(-x))
	
		return
	end subroutine
	!====
    
	!* sigmod������һ�׵���
	subroutine m_df_sigmod( this, index, x, dy )
	implicit none
        class(Sigmod), intent(inout) :: this
		integer, intent(in) :: index
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), intent(out) :: dy
	
		real(PRECISION) :: y
	
		y = 1.0 / (1.0 + exp(-x(index)))
		dy = y * (1 - y)
	
		return
	end subroutine
	!====
	
	!* �������������sigmod������һ�׵���
	subroutine m_df_sigmod_vect( this, x, dy )
	implicit none
        class(Sigmod), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: dy
	
		real(PRECISION), dimension(:), allocatable :: y
	
		allocate( y, source=dy )
	
		y = 1.0 / (1.0 + exp(-x))
		dy = y * (1 - y)
	
        deallocate(y)
        
		return
	end subroutine
	!====

end module