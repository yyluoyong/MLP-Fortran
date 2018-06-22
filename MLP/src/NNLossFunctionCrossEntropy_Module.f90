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
	

    !* �����Ϣ
	subroutine m_print_msg( this )
	implicit none
		class(CrossEntropy), intent(inout) :: this

        write(*, *) "Cross Entropy Function."
        
        return
	end subroutine
	!====
    
    
end module