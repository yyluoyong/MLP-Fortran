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

    procedure, public :: loss  => m_fun_MeanSquareError
    procedure, public :: d_loss => m_df_MeanSquareError
    
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
    
    !* ��ʧ���������һ�㼤����Ա����ĵ���
	!* ���ض�����Ԥ�������ĵ���
	subroutine m_df_MeanSquareError( this, t, r, z, act_fun, dloss )
    use mod_BaseActivationFunction
	implicit none
		class(MeanSquareError), intent(inout) :: this
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

        real(PRECISION), dimension(:), allocatable :: df_to_dr
        
        allocate( df_to_dr, SOURCE=r )
        
        !* df_to_dr Ϊ f'(r)
        call act_fun % df_vect( r, df_to_dr )
        
        dloss = (z - t) * df_to_dr
        
        deallocate( df_to_dr )
        
        return
	end subroutine
	!==== 
    
    !* �����Ϣ
	subroutine m_print_msg( this )
	implicit none
		class(MeanSquareError), intent(inout) :: this

        write(*, *) "Mean Square Error Function."
        
        return
	end subroutine
	!====

end module