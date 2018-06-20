module mod_CrossEntropy
use mod_Precision
use mod_BaseLossFunction
implicit none    

!-------------------
! 工作类：损失函数 |
!-------------------
type, extends(BaseLossFunction), public :: CrossEntropy
    !* 继承自BaseLossFunction并实现其接口

!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: f  => m_fun_CrossEntropy
    procedure, public :: df => m_df_CrossEntropy

end type CrossEntropy
!===================

    !-------------------------
    private :: m_fun_CrossEntropy
    private :: m_df_CrossEntropy
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* CrossEntropy函数
    subroutine m_fun_CrossEntropy( this, t, y, ans )
    implicit none
        class(CrossEntropy), intent(inout) :: this
        !* t 是目标输出向量，对于分类问题，
		!* 它是one-hot编码的向量
		!* y 是网络预测向量
		real(PRECISION), dimension(:), intent(in) :: t
		real(PRECISION), dimension(:), intent(in) :: y
        real(PRECISION), intent(inout) :: ans
    
        ans = -DOT_PRODUCT(t, LOG(y))
    
        return
    end subroutine
    !====
    
	!* CrossEntropy函数的一阶导数
	!* 返回对网络预测向量的导数
	subroutine m_df_CrossEntropy( this, t, y, dy )
	implicit none
        class(CrossEntropy), intent(inout) :: this
		!* t 是目标输出向量，对于分类问题，
		!* 它是one-hot编码的向量
		!* y 是网络预测向量
		real(PRECISION), dimension(:), intent(in) :: t
		real(PRECISION), dimension(:), intent(in) :: y
        real(PRECISION), dimension(:), intent(inout) :: dy
	
		dy = -t / y
	
		return
	end subroutine
	!====
	

end module