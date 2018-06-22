module mod_BaseLossFunction
implicit none
    
!-------------------
! �����ࣺ��ʧ���� |
!-------------------
type, abstract, public :: BaseLossFunction

!||||||||||||    
contains   !|
!||||||||||||

    !* ��ʧ����
    procedure(abs_loss), deferred, public :: loss 
 
    !* ��ʧ���������һ�㼤����Ա����ĵ���
    !* �����PDF�ĵ�
    procedure(abs_d_loss), deferred, public :: d_loss
    
    procedure(abs_print_msg), deferred, public :: print_msg  

end type BaseLossFunction
!===================
    

!-------------------
! �����ࣺ�����ӿ� |
!-------------------	
abstract interface   

	!* ��ʧ����
    !* �ú�����ʱδ�õ�
	subroutine abs_loss( this, t, y, ans )
    use mod_Precision
    import :: BaseLossFunction
	implicit none
		class(BaseLossFunction), intent(inout) :: this
		!* t ��Ŀ�����������y ������Ԥ������
		real(PRECISION), dimension(:), intent(in) :: t
		real(PRECISION), dimension(:), intent(in) :: y
        real(PRECISION), intent(inout) :: ans

	end subroutine
	!====
	
	!* ��ʧ���������һ�㼤����Ա����ĵ���
	!* ���ض�����Ԥ�������ĵ���
	subroutine abs_d_loss( this, t, r, z, act_fun, dloss )
    use mod_Precision
    use mod_BaseActivationFunction
    import :: BaseLossFunction
	implicit none
		class(BaseLossFunction), intent(inout) :: this
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

	end subroutine
	!==== 
    
    
	!* �����Ϣ
	subroutine abs_print_msg( this )
    import :: BaseLossFunction
	implicit none
		class(BaseLossFunction), intent(inout) :: this

	end subroutine
	!====

end interface
!===================
    
end module