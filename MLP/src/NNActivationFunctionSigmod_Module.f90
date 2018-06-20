module mod_Sigmod
use mod_Precision
use mod_BaseActivationFunction
implicit none    

!-------------------
! 工作类：激活函数 |
!-------------------
type, extends(BaseActivationFunction), public :: Sigmod
    !* 继承自BaseActivationFunction并实现其接口

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

    !* sigmod函数
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
    
    !* 接收向量输入的sigmod函数
	subroutine m_fun_sigmod_vect( this, x, y )
	implicit none
        class(Sigmod), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: y
	
		y = 1.0 / (1.0 + exp(-x))
	
		return
	end subroutine
	!====
    
	!* sigmod函数的一阶导数
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
	
	!* 接收向量输入的sigmod函数的一阶导数
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