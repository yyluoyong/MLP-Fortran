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
    procedure(abs_f), deferred, public :: f 
    !* ��ʧ��������
    procedure(abs_df), deferred, public :: df  
    
    procedure(abs_print_msg), deferred, public :: print_msg  

end type BaseLossFunction
!===================
    

!-------------------
! �����ࣺ�����ӿ� |
!-------------------	
abstract interface   

	!* ��ʧ����
	subroutine abs_f( this, t, y, ans )
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
	
	!* ��ʧ����һ�׵���
	!* ���ض�����Ԥ�������ĵ���
	subroutine abs_df( this, t, y, dy )
    use mod_Precision
    import :: BaseLossFunction
	implicit none
		class(BaseLossFunction), intent(inout) :: this
		!* t ��Ŀ�����������y ������Ԥ������
		real(PRECISION), dimension(:), intent(in) :: t
		real(PRECISION), dimension(:), intent(in) :: y
        real(PRECISION), dimension(:), intent(inout) :: dy

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