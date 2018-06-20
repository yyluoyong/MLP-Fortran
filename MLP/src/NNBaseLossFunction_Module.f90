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
    procedure(m_f), deferred, public :: f 
    !* ��ʧ��������
    procedure(m_df), deferred, public :: df  

end type BaseLossFunction
!===================
    

!-------------------
! �����ࣺ�����ӿ� |
!-------------------	
abstract interface   

	!* ��ʧ����
	subroutine m_f( this, t, y, ans )
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
	subroutine m_df( this, t, y, dy )
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

end interface
!===================
    
end module