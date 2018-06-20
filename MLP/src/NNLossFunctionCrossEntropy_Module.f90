module mod_CrossEntropy
use mod_Precision
use mod_BaseLossFunction
implicit none    

!-------------------
! �����ࣺ��ʧ���� |
!-------------------
type, extends(BaseLossFunction), public :: CrossEntropy
    !* �̳���BaseLossFunction��ʵ����ӿ�

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

    !* CrossEntropy����
    subroutine m_fun_CrossEntropy( this, t, y, ans )
    implicit none
        class(CrossEntropy), intent(inout) :: this
        !* t ��Ŀ��������������ڷ������⣬
		!* ����one-hot���������
		!* y ������Ԥ������
		real(PRECISION), dimension(:), intent(in) :: t
		real(PRECISION), dimension(:), intent(in) :: y
        real(PRECISION), intent(inout) :: ans
    
        ans = -DOT_PRODUCT(t, LOG(y))
    
        return
    end subroutine
    !====
    
	!* CrossEntropy������һ�׵���
	!* ���ض�����Ԥ�������ĵ���
	subroutine m_df_CrossEntropy( this, t, y, dy )
	implicit none
        class(CrossEntropy), intent(inout) :: this
		!* t ��Ŀ��������������ڷ������⣬
		!* ����one-hot���������
		!* y ������Ԥ������
		real(PRECISION), dimension(:), intent(in) :: t
		real(PRECISION), dimension(:), intent(in) :: y
        real(PRECISION), dimension(:), intent(inout) :: dy
	
		dy = -t / y
	
		return
	end subroutine
	!====
	

end module