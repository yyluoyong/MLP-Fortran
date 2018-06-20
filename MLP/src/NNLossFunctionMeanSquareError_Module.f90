module mod_MeanSquareError
use mod_Precision
use mod_BaseLossFunction
implicit none    

!-------------------
! �����ࣺ��ʧ���� |
!-------------------
type, extends(BaseLossFunction), public :: MeanSquareError
    !* �̳���BaseLossFunction��ʵ����ӿ�

!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: f  => m_fun_MeanSquareError
    procedure, public :: df => m_df_MeanSquareError

end type MeanSquareError
!===================

    !-------------------------
    private :: m_fun_MeanSquareError
    private :: m_df_MeanSquareError
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* MeanSquareError����
    subroutine m_fun_MeanSquareError( this, t, y, ans )
    implicit none
        class(MeanSquareError), intent(inout) :: this
        !* t ��Ŀ��������������ڷ������⣬
		!* ����one-hot���������
		!* y ������Ԥ������
		real(PRECISION), dimension(:), intent(in) :: t
		real(PRECISION), dimension(:), intent(in) :: y
        real(PRECISION), intent(inout) :: ans
    
        ans = 0.5 * DOT_PRODUCT(y - t, y - t)
    
        return
    end subroutine
    !====
    
	!* MeanSquareError������һ�׵���
	!* ���ض�����Ԥ�������ĵ���
	subroutine m_df_MeanSquareError( this, t, y, dy )
	implicit none
        class(MeanSquareError), intent(inout) :: this
		!* t ��Ŀ��������������ڷ������⣬
		!* ����one-hot���������
		!* y ������Ԥ������
		real(PRECISION), dimension(:), intent(in) :: t
		real(PRECISION), dimension(:), intent(in) :: y
        real(PRECISION), dimension(:), intent(inout) :: dy
	
		dy = y - t
	
		return
	end subroutine
	!====
	

end module