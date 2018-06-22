module mod_CrossEntropy
use mod_Precision
use mod_BaseLossFunction
implicit none    

!-------------------
! 工作类：损失函数 |
!-------------------
type, extends(BaseLossFunction), public :: CrossEntropyWithSoftmax
    !* 继承自BaseLossFunction并实现其接口

!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: loss  => m_fun_CrossEntropy
    procedure, public :: d_loss => m_df_CrossEntropy
    
    procedure, public :: print_msg => m_print_msg

end type CrossEntropyWithSoftmax
!===================

    !-------------------------
    private :: m_fun_CrossEntropy
    private :: m_df_CrossEntropy
    private :: m_print_msg
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* CrossEntropy函数
    subroutine m_fun_CrossEntropy( this, t, y, ans )
    implicit none
        class(CrossEntropyWithSoftmax), intent(inout) :: this
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
    
	!* CrossEntropy损失函数对最后一层激活函数自变量的导数
	!* 返回对网络预测向量的导数
	subroutine m_df_CrossEntropy( this, t, r, z, act_fun, dloss )
    use mod_BaseActivationFunction
	implicit none
		class(CrossEntropyWithSoftmax), intent(inout) :: this
		!* t 是目标输出向量，
        !* r 是最后一层激活函数的自变量，
        !* z 是网络预测向量
        !* act_fun 是最后一层的激活函数，
        !* dloss 是损失函数对 r 的导数
		real(PRECISION), dimension(:), intent(in) :: t
		real(PRECISION), dimension(:), intent(in) :: r
        real(PRECISION), dimension(:), intent(in) :: z
        class(BaseActivationFunction), pointer, intent(in) :: act_fun
        real(PRECISION), dimension(:), intent(inout) :: dloss
        
        dloss = (z - t) 
        
        return
	end subroutine
	!====
	

    !* 输出信息
	subroutine m_print_msg( this )
	implicit none
		class(CrossEntropyWithSoftmax), intent(inout) :: this

        write(*, *) "Cross Entropy Function."
        
        return
	end subroutine
	!====
    
    
end module