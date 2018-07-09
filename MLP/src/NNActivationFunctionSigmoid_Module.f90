module mod_Sigmoid
use mod_Precision
use mod_BaseActivationFunction
implicit none    

!-------------------
! �����ࣺ����� |
!-------------------
type, extends(BaseActivationFunction), public :: Sigmoid
    !* �̳���BaseActivationFunction��ʵ����ӿ�

!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: f       => m_fun_Sigmoid
    procedure, public :: f_vect  => m_fun_Sigmoid_vect 
    procedure, public :: df      => m_df_Sigmoid
    procedure, public :: df_vect => m_df_Sigmoid_vect

end type Sigmoid
!===================

    !-------------------------
    private :: m_fun_Sigmoid
    private :: m_df_Sigmoid
    private :: m_fun_Sigmoid_vect
	private :: m_df_Sigmoid_vect
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* Sigmoid����
    subroutine m_fun_Sigmoid( this, index, x, y )
    implicit none
        class(Sigmoid), intent(inout) :: this
        integer, intent(in) :: index
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), intent(out) :: y
    
        y = 1.0 / (1.0 + exp(-x(index)))
        
        return
    end subroutine
    !====
    
    !* �������������Sigmoid����
	subroutine m_fun_Sigmoid_vect( this, x, y )
	implicit none
        class(Sigmoid), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: y
	
		y = 1.0 / (1.0 + exp(-x))
	
		return
	end subroutine
	!====
    
	!* Sigmoid������һ�׵���
	subroutine m_df_Sigmoid( this, index, x, dy )
	implicit none
        class(Sigmoid), intent(inout) :: this
		integer, intent(in) :: index
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), intent(out) :: dy
	
		real(PRECISION) :: y
	
		y = 1.0 / (1.0 + exp(-x(index)))
		dy = y * (1 - y)
	
		return
	end subroutine
	!====
	
	!* �������������Sigmoid������һ�׵���
	subroutine m_df_Sigmoid_vect( this, x, dy )
	implicit none
        class(Sigmoid), intent(inout) :: this
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