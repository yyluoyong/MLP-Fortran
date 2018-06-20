module mod_Softmax
use mod_Precision
use mod_BaseActivationFunction
implicit none    

!-------------------
! �����ࣺ����� |
!-------------------
type, extends(BaseActivationFunction), public :: Softmax
    !* �̳���BaseActivationFunction��ʵ����ӿ�

!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: f       => m_fun_Softmax
    procedure, public :: f_vect  => m_fun_Softmax_vect 
    procedure, public :: df      => m_df_Softmax
    procedure, public :: df_vect => m_df_Softmax_vect

end type Softmax
!===================

    !-------------------------
    private :: m_fun_Softmax
    private :: m_df_Softmax
    private :: m_fun_Softmax_vect
	private :: m_df_Softmax_vect
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* Softmax����
    subroutine m_fun_Softmax( this, index, x, y )
    implicit none
        class(Softmax), intent(inout) :: this
        integer, intent(in) :: index
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), intent(out) :: y
    
        real(PRECISION) :: sum_exp_x
    
		sum_exp_x = SUM(EXP(x))
		y = EXP(x(index)) / sum_exp_x 
	
        return
    end subroutine
    !====
    
    !* �������������Softmax����
	subroutine m_fun_Softmax_vect( this, x, y )
	implicit none
        class(Softmax), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: y	
	
		real(PRECISION) :: sum_exp_x
		
        sum_exp_x = SUM(EXP(x))
		y = EXP(x) / sum_exp_x 
        
		return
	end subroutine
	!====
    
	!* Softmax������һ�׵���
	subroutine m_df_Softmax( this, index, x, dy  )
	implicit none
        class(Softmax), intent(inout) :: this
		integer, intent(in) :: index
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), intent(out) :: dy
	
		real(PRECISION) :: y, sum_exp_x
	
		sum_exp_x = SUM(EXP(x))
		y = EXP(x(index)) / sum_exp_x 
	
		dy = y * (1 - y)
	
		return
	end subroutine
	!====
	
	!* �������������Softmax������һ�׵���
	subroutine m_df_Softmax_vect( this, x, dy )
	implicit none
        class(Softmax), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: dy
	
		real(PRECISION) :: sum_exp_x
		real(PRECISION), dimension(:), allocatable :: y
	
		allocate(y, SOURCE=x)
	
		sum_exp_x = SUM(EXP(x))
		y = EXP(x) / sum_exp_x 
	
		dy = y * (1 - y)
	
		deallocate(y)
	
		return
	end subroutine
	!====

end module