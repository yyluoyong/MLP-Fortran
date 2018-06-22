module mod_CrossEntropy
use mod_Precision
use mod_BaseLossFunction
implicit none    

!-------------------
! �����ࣺ��ʧ���� |
!-------------------
type, extends(BaseLossFunction), public :: CrossEntropyWithSoftmax
    !* �̳���BaseLossFunction��ʵ����ӿ�

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

    !* CrossEntropy����
    subroutine m_fun_CrossEntropy( this, t, y, ans )
    implicit none
        class(CrossEntropyWithSoftmax), intent(inout) :: this
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
    
	!* CrossEntropy��ʧ���������һ�㼤����Ա����ĵ���
	!* ���ض�����Ԥ�������ĵ���
	subroutine m_df_CrossEntropy( this, t, r, z, act_fun, dloss )
    use mod_BaseActivationFunction
	implicit none
		class(CrossEntropyWithSoftmax), intent(inout) :: this
		!* t ��Ŀ�����������
        !* r �����һ�㼤������Ա�����
        !* z ������Ԥ������
        !* act_fun �����һ��ļ������
        !* dloss ����ʧ������ r �ĵ���
		real(PRECISION), dimension(:), intent(in) :: t
		real(PRECISION), dimension(:), intent(in) :: r
        real(PRECISION), dimension(:), intent(in) :: z
        class(BaseActivationFunction), pointer, intent(in) :: act_fun
        real(PRECISION), dimension(:), intent(inout) :: dloss
        
        dloss = (z - t) 
        
        return
	end subroutine
	!====
	

    !* �����Ϣ
	subroutine m_print_msg( this )
	implicit none
		class(CrossEntropyWithSoftmax), intent(inout) :: this

        write(*, *) "Cross Entropy Function."
        
        return
	end subroutine
	!====
    
    
end module