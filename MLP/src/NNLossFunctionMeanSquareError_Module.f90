module mod_MeanSquareError
use mod_Precision
use mod_BaseLossFunction
implicit none    

!-------------------
! 工作类：损失函数 |
!-------------------
type, extends(BaseLossFunction), public :: MeanSquareError
    !* 继承自BaseLossFunction并实现其接口

!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: f  => m_fun_MeanSquareError
    procedure, public :: df => m_df_MeanSquareError
    
    procedure, public :: print_msg => m_print_msg

end type MeanSquareError
!===================

    !-------------------------
    private :: m_fun_MeanSquareError
    private :: m_df_MeanSquareError
    private :: m_print_msg
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* MeanSquareError函数
    subroutine m_fun_MeanSquareError( this, t, y, ans )
    implicit none
        class(MeanSquareError), intent(inout) :: this
        !* t 是目标输出向量，对于分类问题，
		!* 它是one-hot编码的向量
		!* y 是网络预测向量
		real(PRECISION), dimension(:), intent(in) :: t
		real(PRECISION), dimension(:), intent(in) :: y
        real(PRECISION), intent(inout) :: ans
    
        ans = 0.5 * DOT_PRODUCT(y - t, y - t)
    
        return
    end subroutine
    !====
    
	!* MeanSquareError函数的一阶导数
	!* 返回对网络预测向量的导数
	subroutine m_df_MeanSquareError( this, t, y, dy )
	implicit none
        class(MeanSquareError), intent(inout) :: this
		!* t 是目标输出向量，对于分类问题，
		!* 它是one-hot编码的向量
		!* y 是网络预测向量
		real(PRECISION), dimension(:), intent(in) :: t
		real(PRECISION), dimension(:), intent(in) :: y
        real(PRECISION), dimension(:), intent(inout) :: dy
	
		dy = y - t
	
		return
	end subroutine
	!====
	
    
    !* 输出信息
	subroutine m_print_msg( this )
	implicit none
		class(MeanSquareError), intent(inout) :: this

        write(*, *) "Mean Square Error Function."
        
        return
	end subroutine
	!====

end module