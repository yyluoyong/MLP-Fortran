module mod_Linear
use mod_Precision
use mod_BaseActivationFunction
implicit none    

!-------------------
! 工作类：激活函数 |
!-------------------
type, extends(BaseActivationFunction), public :: Linear
    !* 继承自BaseActivationFunction并实现其接口

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

    !* Linear函数
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
    
    !* 接收向量输入的Linear函数
	subroutine m_fun_Linear_vect( this, x, y )
	implicit none
        class(Linear), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: y	
	
        y = x
        
		return
	end subroutine
	!====
    
	!* Linear函数的一阶导数
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
	
	!* 接收向量输入的Linear函数的一阶导数
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