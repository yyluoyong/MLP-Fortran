module mod_ELU
use mod_Precision
use mod_BaseActivationFunction
implicit none    

	!* undo: 从文件中读取 ELU 的参数 
	real(PRECISION), parameter :: ELU_ALPHA = 0.3

!-------------------
! 工作类：激活函数 |
!-------------------
type, extends(BaseActivationFunction), public :: ELU
    !* 继承自BaseActivationFunction并实现其接口

!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: f       => m_fun_ELU
    procedure, public :: f_vect  => m_fun_ELU_vect 
    procedure, public :: df      => m_df_ELU
    procedure, public :: df_vect => m_df_ELU_vect

end type ELU
!===================

    !-------------------------
    private :: m_fun_ELU
    private :: m_df_ELU
    private :: m_fun_ELU_vect
	private :: m_df_ELU_vect
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* ELU函数
    subroutine m_fun_ELU( this, x, y )
    implicit none
        class(ELU), intent(inout) :: this
        real(PRECISION), intent(in) :: x
		real(PRECISION), intent(out) :: y
    
        if (x < 0) then
            y = ELU_ALPHA * (exp(x) - 1)
        else
            y = x
        end if
    
        return
    end subroutine
    !====
    
    !* 接收向量输入的ELU函数
	subroutine m_fun_ELU_vect( this, x, y )
	implicit none
        class(ELU), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: y
	
		integer :: count, i
	
        count = SIZE(X)
        
        do i=1, count
            if (x(i) < 0) then
                y(i) = ELU_ALPHA * (exp(x(i)) - 1)
            else
                y(i) = x(i)
            end if
        end do
        
		return
	end subroutine
	!====
    
	!* ELU函数的一阶导数
	subroutine m_df_ELU( this, x, dy )
	implicit none
        class(ELU), intent(inout) :: this
		real(PRECISION), intent(in) :: x
		real(PRECISION), intent(out) :: dy
	
		if (x < 0) then
            dy = ELU_ALPHA * exp(x)
        else
            dy = 1
        end if
	
		return
	end subroutine
	!====
	
	!* 接收向量输入的ELU函数的一阶导数
	subroutine m_df_ELU_vect( this, x, dy )
	implicit none
        class(ELU), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: dy
	
		integer :: count, i
	
        count = SIZE(X)
        
        do i=1, count
            if (x(i) < 0) then
                dy(i) = ELU_ALPHA * exp(x(i))
            else
                dy(i) = 1
            end if
        end do
	
		return
	end subroutine
	!====

end module