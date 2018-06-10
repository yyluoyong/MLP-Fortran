module mod_TanH
use mod_Precision
use mod_BaseActivationFunction
implicit none    

!----------------
! 工作类：激活函数 |
!----------------
type, extends(BaseActivationFunction), public :: Tan_H
    !* 继承自BaseActivationFunction并实现其接口

!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: f       => m_fun_tanh
    procedure, public :: f_vect  => m_fun_tanh_vect 
    procedure, public :: df      => m_df_tanh
    procedure, public :: df_vect => m_df_tanh_vect

end type Tan_H
!===================

    !-------------------------
    private :: m_fun_tanh
    private :: m_df_tanh
    private :: m_fun_tanh_vect
	private :: m_df_tanh_vect
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* tanh函数
    subroutine m_fun_tanh( this, x, y )
    implicit none
        class(Tan_H), intent(inout) :: this
        real(PRECISION), intent(in) :: x
		real(PRECISION), intent(out) :: y
    
        y = TANH(x)
        
        return
    end subroutine
    !====
    
    !* 接收向量输入的tanh函数
	subroutine m_fun_tanh_vect( this, x, y )
	implicit none
        class(Tan_H), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: y
	
		y = TANH(x)
	
		return
	end subroutine
	!====
    
	!* tanh函数的一阶导数
	subroutine m_df_tanh( this, x, dy )
	implicit none
        class(Tan_H), intent(inout) :: this
		real(PRECISION), intent(in) :: x
		real(PRECISION), intent(out) :: dy
	
		real(PRECISION) :: y
	
		y = TANH(x)
		dy = 1 - y*y
	
		return
	end subroutine
	!====
	
	!* 接收向量输入的tanh函数的一阶导数
	subroutine m_df_tanh_vect( this, x, dy )
	implicit none
        class(Tan_H), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: dy
	
		real(PRECISION), dimension(:), allocatable :: y
	
		allocate( y, source=dy )
	
		y = TANH(x)
		dy = 1 - y*y
        
        deallocate(y)
	
		return
	end subroutine
	!====

end module