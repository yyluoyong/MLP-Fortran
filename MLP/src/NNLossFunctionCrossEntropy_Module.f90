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
    
    procedure, public :: print_msg => m_print_msg

end type CrossEntropy
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
	
        integer :: count
        integer :: i
        
        count = SIZE(t)
        
        do i=1, count
            if (abs(t(i)) < 1.E-16) then
                dy(i) = 0
            else
		        dy(i) = -t(i) / y(i)
            end if
        end do
        write(*, *) minval(dy), maxval(dy)
	
		return
	end subroutine
	!====
	

    !* 输出信息
	subroutine m_print_msg( this )
	implicit none
		class(CrossEntropy), intent(inout) :: this

        write(*, *) "Cross Entropy Function."
        
        return
	end subroutine
	!====
    
    
end module